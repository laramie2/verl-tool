import argparse
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class TopKLogprobsRequest(BaseModel):
    input_ids: list[list[int]]
    attention_mask: list[list[int]] | None = None
    response_length: int
    topk: int | None = None
    temperature: float = 1.0
    multi_modal_inputs: list[dict[str, Any] | None] | None = None


class TeacherEngine:
    def __init__(self, model_path: str, dtype: str, default_topk: int):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        self.default_topk = default_topk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype, torch.bfloat16)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if type(config) in AutoModelForVision2Seq._model_mapping.keys():
            model_cls = AutoModelForVision2Seq
        else:
            model_cls = AutoModelForCausalLM

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = torch_dtype
        else:
            kwargs["torch_dtype"] = "auto"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            kwargs["device_map"] = "auto"
            self.model = model_cls.from_pretrained(model_path, **kwargs)
        else:
            self.model = model_cls.from_pretrained(model_path, **kwargs).to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

    def _tensor_from_payload(self, key: str, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            dtype = torch.long if key.endswith("ids") or key.endswith("mask") or "grid" in key else torch.float32
            tensor = torch.tensor(value, dtype=dtype)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=self.model_dtype)
        return tensor.to(self.device)

    def _prepare_multi_modal_inputs(
        self, multi_modal_inputs: list[dict[str, Any] | None] | None
    ) -> dict[str, torch.Tensor]:
        if not multi_modal_inputs:
            return {}

        collected: dict[str, list[torch.Tensor]] = {}
        for item in multi_modal_inputs:
            if not item:
                continue
            for key, value in item.items():
                if value is None:
                    continue
                collected.setdefault(key, []).append(self._tensor_from_payload(key, value))

        return {key: torch.cat(values, dim=0) for key, values in collected.items() if values}

    @torch.inference_mode()
    def topk_logprobs(self, request: TopKLogprobsRequest) -> dict[str, Any]:
        input_ids = torch.tensor(request.input_ids, dtype=torch.long, device=self.device)
        if request.attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.long, device=self.device)

        response_length = int(request.response_length)
        if response_length <= 0:
            raise ValueError("response_length must be positive")
        if input_ids.shape[1] < response_length + 1:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} is too short for response_length={response_length}"
            )

        topk = int(request.topk or self.default_topk)
        temperature = float(request.temperature or 1.0)
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        multi_modal_inputs = self._prepare_multi_modal_inputs(request.multi_modal_inputs)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **multi_modal_inputs,
        )
        logits = outputs.logits[:, -response_length - 1 : -1, :].float() / temperature
        logprobs = torch.log_softmax(logits, dim=-1)
        topk_logprobs, topk_ids = torch.topk(logprobs, k=topk, dim=-1)
        return {
            "topk_ids": topk_ids.cpu().tolist(),
            "topk_logprobs": topk_logprobs.cpu().tolist(),
        }


def build_app(engine: TeacherEngine) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/topk_logprobs")
    def topk_logprobs(request: TopKLogprobsRequest):
        return engine.topk_logprobs(request)

    return app


def main():
    parser = argparse.ArgumentParser(description="OPD teacher-forcing top-k logprob server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    if args.tensor_parallel_size != 1:
        print(
            "opd_teacher_server uses transformers device_map=auto rather than vLLM tensor parallelism; "
            f"tensor_parallel_size={args.tensor_parallel_size} is treated as a placement hint.",
            flush=True,
        )

    engine = TeacherEngine(model_path=args.model, dtype=args.dtype, default_topk=args.topk)
    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
