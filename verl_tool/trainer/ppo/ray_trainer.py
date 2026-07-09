import gc
import ray
import uuid
import torch
import os
import json
import time
import math
import numpy as np
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
from typing import Optional
from omegaconf import OmegaConf
from tqdm import tqdm
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    process_validation_metrics,
) # for train and validate
from verl.trainer.ppo.ray_trainer import (
    DataProto,
) # for init
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import Tracking


_REWARD_EXTRA_METRIC_PREFIXES = ("wiki_", "browser_")
_REWARD_EXTRA_SKIP_KEYS = {"turn_reward"}
STEP_GC_EACH_STEP = os.getenv("VERLTOOL_GC_EACH_STEP", "1").strip().lower() in {"1", "true", "yes", "on"}
STEP_EMPTY_CUDA_CACHE = os.getenv("VERLTOOL_EMPTY_CUDA_CACHE_EACH_STEP", "0").strip().lower() in {"1", "true", "yes", "on"}


def _cleanup_step_memory() -> None:
    if STEP_GC_EACH_STEP:
        gc.collect()
    if STEP_EMPTY_CUDA_CACHE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _update_reward_extra_metrics(metrics: dict, reward_extra_infos_dict: dict | None) -> None:
    """Expose scalar reward-manager diagnostics in train step console/wandb logs."""
    if not reward_extra_infos_dict:
        return
    for key, values in reward_extra_infos_dict.items():
        if key in _REWARD_EXTRA_SKIP_KEYS or not key.startswith(_REWARD_EXTRA_METRIC_PREFIXES):
            continue
        numeric_values = []
        for value in values:
            if isinstance(value, np.ndarray):
                if value.size != 1:
                    continue
                value = value.item()
            elif isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (bool, int, float)):
                numeric_values.append(float(value))
        if not numeric_values:
            continue
        arr = np.asarray(numeric_values, dtype=np.float64)
        metrics[f"reward_extra/{key}/mean"] = float(arr.mean())
        metrics[f"reward_extra/{key}/max"] = float(arr.max())
        metrics[f"reward_extra/{key}/min"] = float(arr.min())


##############################################################################
#### Replace the original classes/functions with verl-tool customized ones ####
import verl.experimental.agent_loop
from verl_tool.agent_loop import AgentLoopManager
import verl.trainer.ppo.ray_trainer
from .reward import compute_reward, compute_reward_async
from verl_tool.workers.rollout.vllm_rollout.vllm_async_server import VerlToolvLLMHttpServer
import verl.workers.rollout.vllm_rollout.vllm_async_server
from .metric_util import compute_data_metrics, process_validation_metrics
verl.experimental.agent_loop.AgentLoopManager = AgentLoopManager
verl.trainer.ppo.ray_trainer.compute_reward = compute_reward
verl.trainer.ppo.ray_trainer.compute_reward_async = compute_reward_async
verl.trainer.ppo.ray_trainer.compute_data_metrics = compute_data_metrics
verl.trainer.ppo.ray_trainer.process_validation_metrics = process_validation_metrics
verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer = VerlToolvLLMHttpServer
##############################################################################

class AgentRayPPOTrainer(RayPPOTrainer):
    def _select_model_forward_batch(self, batch: DataProto) -> DataProto:
        """Keep heavyweight agent trajectory metadata off model worker RPCs."""
        model_forward_non_tensor_keys = []
        if "multi_modal_inputs" in batch.non_tensor_batch:
            model_forward_non_tensor_keys.append("multi_modal_inputs")
        return batch.select(
            batch_keys=list(batch.batch.keys()),
            non_tensor_batch_keys=model_forward_non_tensor_keys,
        )

    def _compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        model_forward_batch = self._select_model_forward_batch(batch)

        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(model_forward_batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics.update({"actor/entropy": entropy_agg.detach().item()})
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                from verl.utils.debug.metrics import calculate_debug_metrics

                metrics.update(calculate_debug_metrics(batch))

        if self.use_reference_policy:
            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(model_forward_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(model_forward_batch)
                batch = batch.union(ref_log_prob)

        return batch

    @staticmethod
    def _post_json(url: str, payload: dict, timeout: int) -> dict:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _jsonify_opd_value(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: AgentRayPPOTrainer._jsonify_opd_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [AgentRayPPOTrainer._jsonify_opd_value(val) for val in value]
        return value

    def _prepare_opd_reason_mask(self, batch: DataProto, metrics: dict) -> Optional[torch.Tensor]:
        response_mask = batch.batch["response_mask"].float()
        bsz, response_len = response_mask.shape
        reason_mask = torch.zeros((bsz, response_len), dtype=torch.float32)

        raw_masks = batch.non_tensor_batch.get("opd_reason_mask", None)
        if raw_masks is None:
            metrics["actor/opd_reason_tokens"] = 0.0
            return None

        for row_idx, raw_mask in enumerate(raw_masks):
            if raw_mask is None:
                continue
            if isinstance(raw_mask, np.ndarray):
                raw_mask = raw_mask.tolist()
            raw_mask = list(raw_mask)
            keep_len = min(len(raw_mask), response_len)
            if keep_len > 0:
                reason_mask[row_idx, :keep_len] = torch.tensor(raw_mask[:keep_len], dtype=torch.float32)

        reason_mask = reason_mask * response_mask.cpu()
        metrics["actor/opd_reason_tokens"] = float(reason_mask.sum().item())
        if reason_mask.sum().item() <= 0:
            return None
        return reason_mask

    def _attach_opd_teacher_logits(self, batch: DataProto, metrics: dict) -> DataProto:
        actor_cfg = self.config.actor_rollout_ref.actor
        if not actor_cfg.get("opd_enable", False):
            return batch

        reason_mask = self._prepare_opd_reason_mask(batch, metrics)
        if reason_mask is None:
            metrics["actor/opd_teacher/requests"] = 0.0
            return batch

        teacher_url = actor_cfg.get("opd_teacher_url", "")
        if not teacher_url:
            raise ValueError("actor_rollout_ref.actor.opd_teacher_url must be set when opd_enable=True")

        input_ids = batch.batch["input_ids"].cpu()
        attention_mask = batch.batch["attention_mask"].cpu()
        response_len = int(batch.batch["responses"].shape[1])
        topk = int(actor_cfg.get("opd_teacher_topk", 20))
        teacher_batch_size = max(int(actor_cfg.get("opd_teacher_batch_size", 1)), 1)
        timeout = int(actor_cfg.get("opd_teacher_timeout", 600))
        teacher_temperature = float(actor_cfg.get("opd_teacher_temperature", 1.0))

        all_topk_ids = []
        all_topk_logprobs = []
        endpoint = teacher_url.rstrip("/") + "/topk_logprobs"
        multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs", None)
        num_requests = 0
        for start in range(0, input_ids.shape[0], teacher_batch_size):
            end = min(start + teacher_batch_size, input_ids.shape[0])
            payload = {
                "input_ids": input_ids[start:end].tolist(),
                "attention_mask": attention_mask[start:end].tolist(),
                "response_length": response_len,
                "topk": topk,
                "temperature": teacher_temperature,
            }
            if multi_modal_inputs is not None:
                payload["multi_modal_inputs"] = [
                    self._jsonify_opd_value(item) for item in multi_modal_inputs[start:end]
                ]
            result = self._post_json(endpoint, payload, timeout=timeout)
            all_topk_ids.append(torch.tensor(result["topk_ids"], dtype=torch.long))
            all_topk_logprobs.append(torch.tensor(result["topk_logprobs"], dtype=torch.float32))
            num_requests += 1

        teacher_topk_ids = torch.cat(all_topk_ids, dim=0)
        teacher_topk_logprobs = torch.cat(all_topk_logprobs, dim=0)
        expected_shape = (input_ids.shape[0], response_len, topk)
        if tuple(teacher_topk_ids.shape) != expected_shape:
            raise ValueError(f"Unexpected OPD teacher topk shape: {tuple(teacher_topk_ids.shape)} != {expected_shape}")

        batch.batch["opd_reason_mask"] = reason_mask.to(batch.batch["response_mask"].device)
        batch.batch["opd_teacher_topk_ids"] = teacher_topk_ids.to(batch.batch["responses"].device)
        batch.batch["opd_teacher_topk_logprobs"] = teacher_topk_logprobs.to(batch.batch["response_mask"].device)
        metrics["actor/opd_teacher/requests"] = float(num_requests)
        metrics["actor/opd_teacher/topk"] = float(topk)
        return batch

    def _collect_opd_teacher_logits_or_fallback(self, batch: DataProto, metrics: dict) -> DataProto:
        actor_cfg = self.config.actor_rollout_ref.actor
        if not actor_cfg.get("opd_enable", False):
            return batch

        reason_mask = self._prepare_opd_reason_mask(batch, metrics)
        if reason_mask is None:
            metrics["actor/opd_teacher/requests"] = 0.0
            return batch

        teacher_url = actor_cfg.get("opd_teacher_url", "")
        if not teacher_url:
            raise ValueError("actor_rollout_ref.actor.opd_teacher_url must be set when opd_enable=True")

        traj_ids = batch.non_tensor_batch.get("opd_traj_id", None)
        submitted = batch.non_tensor_batch.get("opd_teacher_submitted", None)

        response_len = int(batch.batch["responses"].shape[1])
        topk = int(actor_cfg.get("opd_teacher_topk", 20))
        teacher_temperature = float(actor_cfg.get("opd_teacher_temperature", 1.0))
        submit_endpoint = teacher_url.rstrip("/") + "/submit_topk_logprobs"
        input_ids = batch.batch["input_ids"].cpu()
        attention_mask = batch.batch["attention_mask"].cpu()
        multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs", None)

        def submit_row(idx: int, request_id: str):
            payload = {
                "request_id": request_id,
                "input_ids": [input_ids[idx].tolist()],
                "attention_mask": [attention_mask[idx].tolist()],
                "response_length": response_len,
                "topk": topk,
                "temperature": teacher_temperature,
                "reason_mask": [reason_mask[idx].cpu().tolist()],
            }
            if multi_modal_inputs is not None:
                payload["multi_modal_inputs"] = [self._jsonify_opd_value(multi_modal_inputs[idx])]
            self._post_json(submit_endpoint, payload, timeout=10)

        active_rows = [idx for idx in range(reason_mask.shape[0]) if reason_mask[idx].sum().item() > 0]
        request_ids: list[str] = []
        row_by_request_id: dict[str, int] = {}
        late_submits = 0
        for idx in active_rows:
            request_id = str(traj_ids[idx]) if traj_ids is not None and traj_ids[idx] is not None else str(uuid.uuid4())
            is_submitted = bool(submitted[idx]) if submitted is not None and submitted[idx] is not None else False
            if not is_submitted:
                submit_row(idx, request_id)
                late_submits += 1
            request_ids.append(request_id)
            row_by_request_id[request_id] = idx

        metrics["actor/opd_teacher/late_submits"] = float(late_submits)
        if not request_ids:
            batch.batch["opd_reason_mask"] = reason_mask.to(batch.batch["response_mask"].device)
            batch.batch["opd_teacher_topk_ids"] = torch.zeros(
                (len(batch), response_len, topk), dtype=torch.long, device=batch.batch["responses"].device
            )
            batch.batch["opd_teacher_topk_logprobs"] = torch.zeros(
                (len(batch), response_len, topk), dtype=torch.float32, device=batch.batch["response_mask"].device
            )
            metrics["actor/opd_teacher/requests"] = 0.0
            metrics["actor/opd_teacher/topk"] = float(topk)
            return batch

        endpoint = teacher_url.rstrip("/") + "/fetch_topk_logprobs"
        per_request_timeout = int(actor_cfg.get("opd_teacher_timeout", 600))
        collect_timeout = max(per_request_timeout, per_request_timeout * max(len(request_ids), 1))
        poll_interval = float(actor_cfg.get("opd_teacher_poll_interval", 2.0))
        deadline = time.monotonic() + collect_timeout
        pending = set(request_ids)
        ready: dict[str, dict] = {}

        while pending:
            result = self._post_json(
                endpoint,
                {"request_ids": sorted(pending), "pop": False},
                timeout=min(max(per_request_timeout, 1), 60),
            )
            records = result.get("results", {})
            for request_id, record in records.items():
                status = record.get("status")
                if status == "done":
                    ready[request_id] = record["result"]
                    pending.discard(request_id)
                elif status == "error":
                    raise RuntimeError(f"OPD teacher job {request_id} failed: {record.get('error')}")
                elif status == "missing":
                    row_idx = row_by_request_id.get(request_id)
                    if row_idx is None:
                        raise RuntimeError(f"OPD teacher job {request_id} is missing and cannot be resubmitted")
                    submit_row(row_idx, request_id)
                    metrics["actor/opd_teacher/resubmits"] = metrics.get("actor/opd_teacher/resubmits", 0.0) + 1.0

            if pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for {len(pending)} OPD teacher jobs")
                time.sleep(poll_interval)

        self._post_json(endpoint, {"request_ids": request_ids, "pop": True}, timeout=min(max(per_request_timeout, 1), 60))

        teacher_topk_ids = torch.zeros((len(batch), response_len, topk), dtype=torch.long)
        teacher_topk_logprobs = torch.zeros((len(batch), response_len, topk), dtype=torch.float32)
        for request_id, teacher_result in ready.items():
            row_idx = row_by_request_id[request_id]
            row_topk_ids = torch.tensor(teacher_result["topk_ids"], dtype=torch.long)
            row_topk_logprobs = torch.tensor(teacher_result["topk_logprobs"], dtype=torch.float32)
            if row_topk_ids.dim() == 3:
                row_topk_ids = row_topk_ids.squeeze(0)
            if row_topk_logprobs.dim() == 3:
                row_topk_logprobs = row_topk_logprobs.squeeze(0)
            expected = (response_len, topk)
            if tuple(row_topk_ids.shape) != expected:
                raise ValueError(f"Unexpected OPD teacher topk shape for {request_id}: {tuple(row_topk_ids.shape)} != {expected}")
            teacher_topk_ids[row_idx] = row_topk_ids
            teacher_topk_logprobs[row_idx] = row_topk_logprobs

        batch.batch["opd_reason_mask"] = reason_mask.to(batch.batch["response_mask"].device)
        batch.batch["opd_teacher_topk_ids"] = teacher_topk_ids.to(batch.batch["responses"].device)
        batch.batch["opd_teacher_topk_logprobs"] = teacher_topk_logprobs.to(batch.batch["response_mask"].device)
        metrics["actor/opd_teacher/requests"] = float(len(request_ids))
        metrics["actor/opd_teacher/topk"] = float(topk)
        metrics["actor/opd_teacher/fallback_sync"] = 0.0
        metrics["actor/opd_teacher/pending_after_collect"] = 0.0
        return batch

    def _filter_and_accumulate_dapo_batch(
        self,
        new_batch: DataProto,
        batch: Optional[DataProto],
        num_prompt_in_batch: int,
        num_gen_batches: int,
        metrics: dict,
    ) -> tuple[Optional[DataProto], int, bool]:
        filter_groups_cfg = self.config.algorithm.get("filter_groups", None)
        if filter_groups_cfg is None or not filter_groups_cfg.get("enable", False):
            return new_batch, len(new_batch.non_tensor_batch["uid"]) // self.config.actor_rollout_ref.rollout.n, True

        metric_name = filter_groups_cfg.metric
        if metric_name == "seq_final_reward":
            new_batch.non_tensor_batch["seq_final_reward"] = (
                new_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
            )
        elif metric_name == "seq_reward":
            new_batch.non_tensor_batch["seq_reward"] = (
                new_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
            )
        elif metric_name in new_batch.batch:
            new_batch.non_tensor_batch[metric_name] = new_batch.batch[metric_name].detach().cpu().numpy()
        elif metric_name not in new_batch.non_tensor_batch:
            raise KeyError(
                f"algorithm.filter_groups.metric={metric_name!r} is not available in batch or non_tensor_batch."
            )

        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True):
            prompt_uid2metric_vals[uid].append(metric_val)

        prompt_uid2metric_std = {
            prompt_uid: np.std(metric_vals) for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
        }
        kept_prompt_uids = [
            uid
            for uid, std in prompt_uid2metric_std.items()
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]
        num_prompt_in_batch += len(kept_prompt_uids)

        kept_traj_idxs = [
            idx
            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"])
            if traj_from_prompt_uid in kept_prompt_uids
        ]
        if kept_traj_idxs:
            new_batch = new_batch[kept_traj_idxs]
            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

        total_prompts = len(prompt_uid2metric_vals)
        metrics["train/dapo_filter/kept_prompts_this_gen"] = len(kept_prompt_uids)
        metrics["train/dapo_filter/total_prompts_this_gen"] = total_prompts
        metrics["train/dapo_filter/kept_ratio_this_gen"] = (
            len(kept_prompt_uids) / total_prompts if total_prompts > 0 else 0.0
        )

        prompt_bsz = self.config.data.train_batch_size
        metrics["train/dapo_filter/kept_prompts_accumulated"] = num_prompt_in_batch
        metrics["train/dapo_filter/target_prompts"] = prompt_bsz
        if num_prompt_in_batch < prompt_bsz:
            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
            max_num_gen_batches = filter_groups_cfg.max_num_gen_batches
            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                print(f"{num_gen_batches=}. Keep generating...")
                return batch, num_prompt_in_batch, False
            raise ValueError(
                f"{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. "
                "Please check if your data are too difficult. You could also set "
                "algorithm.filter_groups.max_num_gen_batches=0 to enable endless trials."
            )

        if batch is None:
            raise ValueError("DAPO filter_groups kept no trajectories but reached the prompt batch threshold.")

        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        return batch[:traj_bsz], num_prompt_in_batch, True

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        save_best_by_metric = self.config.trainer.get("save_best_by_metric", False)
        best_checkpoint_metric_key = None
        best_checkpoint_metric_value = None
        best_checkpoint_step = None
        last_validation_step = None
        last_completed_step = None
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            if save_best_by_metric:
                best_checkpoint_metric_key, best_checkpoint_metric_value = self._get_best_checkpoint_metric(val_metrics)
                best_checkpoint_step = self.global_steps if best_checkpoint_metric_key is not None else None
                last_validation_step = self.global_steps
                if best_checkpoint_metric_key is not None:
                    print(
                        "Initial best checkpoint metric baseline: "
                        f"{best_checkpoint_metric_key}={best_checkpoint_metric_value} at step {best_checkpoint_step}"
                    )
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        self.max_steps_duration = 0
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                gen_batch = None
                gen_batch_output = None
                gen_baseline_batch = None
                gen_baseline_output = None
                new_batch = None
                model_forward_batch = None
                reward_tensor = None
                reward_extra_infos_dict = None
                reward_baseline_tensor = None
                rm_scores = None
                values = None
                critic_output = None
                actor_output = None

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(new_batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["epoch"] = epoch
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            new_batch = new_batch.union(gen_baseline_output)

                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    if self.config.algorithm.use_kl_in_reward:
                        new_batch = self._compute_kl_related_metrics(new_batch, metrics, timing_raw)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=new_batch, config=self.config, tokenizer=self.tokenizer
                            )
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            _update_reward_extra_metrics(metrics, reward_extra_infos_dict)

                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    batch, num_prompt_in_batch, ready_to_update = self._filter_and_accumulate_dapo_batch(
                        new_batch=new_batch,
                        batch=batch,
                        num_prompt_in_batch=num_prompt_in_batch,
                        num_gen_batches=num_gen_batches,
                        metrics=metrics,
                    )
                    if not ready_to_update:
                        self.gen_steps += 1
                        new_batch = None
                        gen_batch = None
                        gen_batch_output = None
                        reward_tensor = None
                        reward_extra_infos_dict = None
                        rm_scores = None
                        _cleanup_step_memory()
                        continue

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self._compute_kl_related_metrics(batch, metrics, timing_raw)

                    model_forward_batch = self._select_model_forward_batch(batch)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(model_forward_batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        if self.config.actor_rollout_ref.actor.get("opd_enable", False):
                            with marked_timer("opd_teacher_wait", timing_raw, color="orange"):
                                batch = self._collect_opd_teacher_logits_or_fallback(batch, metrics)

                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        rollout_extra_infos = {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in batch.non_tensor_batch.items()
                            if key not in {"uid", "data_source", "reward_model", "raw_prompt_ids", "multi_modal_data"}
                        }
                        self._log_rollout_data(batch, rollout_extra_infos, timing_raw, rollout_data_dir)

                save_due_to_best_metric = False

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        last_validation_step = self.global_steps
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                    if save_best_by_metric and val_metrics:
                        metric_key, metric_value = self._get_best_checkpoint_metric(val_metrics)
                        if metric_key is not None:
                            improved = self._is_better_checkpoint_metric(metric_value, best_checkpoint_metric_value)
                            metrics["checkpoint/best_metric_current"] = metric_value
                            metrics["checkpoint/best_metric_value"] = (
                                metric_value
                                if improved
                                else (
                                    best_checkpoint_metric_value
                                    if best_checkpoint_metric_value is not None
                                    else float("nan")
                                )
                            )
                            metrics["checkpoint/best_metric_improved"] = float(improved)
                            if improved:
                                best_checkpoint_metric_key = metric_key
                                best_checkpoint_metric_value = metric_value
                                best_checkpoint_step = self.global_steps
                                save_due_to_best_metric = True
                                print(
                                    "New best checkpoint metric: "
                                    f"{metric_key}={metric_value} at step {self.global_steps}"
                                )
                            else:
                                print(
                                    "Checkpoint metric did not improve: "
                                    f"{metric_key}={metric_value}, best={best_checkpoint_metric_value} "
                                    f"at step {best_checkpoint_step}"
                                )

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                save_due_to_freq = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                )
                save_due_to_last = is_last_step and self.config.trainer.get("save_last", False)
                save_due_to_esi = esi_close_to_expiration
                should_save_checkpoint = (
                    save_due_to_best_metric or save_due_to_freq or save_due_to_last or save_due_to_esi
                )
                if should_save_checkpoint:
                    save_reasons = []
                    if save_due_to_best_metric:
                        save_reasons.append("best_metric")
                    if save_due_to_freq:
                        save_reasons.append("save_freq")
                    if save_due_to_last:
                        save_reasons.append("last_step")
                    if save_due_to_esi:
                        save_reasons.append("esi_expiration")
                    print(f"Saving checkpoint at step {self.global_steps}; reasons={','.join(save_reasons)}")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "train/num_gen_batches": num_gen_batches,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                last_completed_step = self.global_steps
                self.global_steps += 1
                self.gen_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

                # Break references to large per-step DataProto/TensorDict objects before
                # the next rollout starts. This helps Ray avoid crossing its node memory
                # kill threshold when rollout outputs, reward inputs, and update batches
                # briefly coexist on the driver.
                new_batch = None
                gen_batch = None
                gen_batch_output = None
                gen_baseline_batch = None
                gen_baseline_output = None
                model_forward_batch = None
                reward_tensor = None
                reward_extra_infos_dict = None
                reward_baseline_tensor = None
                rm_scores = None
                values = None
                critic_output = None
                actor_output = None

                timing_raw = defaultdict(float)
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                _cleanup_step_memory()

        if (
            last_completed_step is not None
            and self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and last_validation_step != last_completed_step
        ):
            original_global_steps = self.global_steps
            self.global_steps = last_completed_step
            final_metrics = {"training/global_step": self.global_steps}
            final_timing_raw = defaultdict(float)

            try:
                with marked_timer("final_testing", final_timing_raw, color="green"):
                    final_val_metrics: dict = self._validate()
                last_validation_step = self.global_steps
                final_metrics.update(final_val_metrics)

                save_due_to_best_metric = False
                if save_best_by_metric and final_val_metrics:
                    metric_key, metric_value = self._get_best_checkpoint_metric(final_val_metrics)
                    if metric_key is not None:
                        improved = self._is_better_checkpoint_metric(metric_value, best_checkpoint_metric_value)
                        final_metrics["checkpoint/best_metric_current"] = metric_value
                        final_metrics["checkpoint/best_metric_value"] = (
                            metric_value
                            if improved
                            else (
                                best_checkpoint_metric_value
                                if best_checkpoint_metric_value is not None
                                else float("nan")
                            )
                        )
                        final_metrics["checkpoint/best_metric_improved"] = float(improved)
                        if improved:
                            best_checkpoint_metric_key = metric_key
                            best_checkpoint_metric_value = metric_value
                            best_checkpoint_step = self.global_steps
                            save_due_to_best_metric = True
                            print(
                                "New best checkpoint metric at final validation: "
                                f"{metric_key}={metric_value} at step {self.global_steps}"
                            )
                        else:
                            print(
                                "Final checkpoint metric did not improve: "
                                f"{metric_key}={metric_value}, best={best_checkpoint_metric_value} "
                                f"at step {best_checkpoint_step}"
                            )

                if save_due_to_best_metric:
                    print(f"Saving checkpoint at step {self.global_steps}; reasons=best_metric,final_validation")
                    with marked_timer("save_checkpoint", final_timing_raw, color="green"):
                        self._save_checkpoint()

                logger.log(data=final_metrics, step=self.global_steps)
                pprint(f"Final validation metrics: {final_val_metrics}")
            finally:
                self.global_steps = original_global_steps

        progress_bar.close()

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_attention_mask = test_output_gen_batch.batch["attention_mask"][:, test_output_gen_batch.batch["prompts"].shape[1]:]
            output_texts = [self.tokenizer.decode(ids[output_attention_mask[i]==1], skip_special_tokens=False) for i, ids in enumerate(output_ids)]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    
            tool_interact_info = test_batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                if "tool_interact_info" not in reward_extra_infos_dict:
                    reward_extra_infos_dict["tool_interact_info"] = []
                if "traj_stop_reason" not in reward_extra_infos_dict:
                    reward_extra_infos_dict["traj_stop_reason"] = []
                reward_extra_infos_dict["tool_interact_info"].extend(tool_interact_info)
                reward_extra_infos_dict["traj_stop_reason"].extend(
                    test_batch.non_tensor_batch.get("traj_stop_reason", [None] * reward_tensor.shape[0])
                )
                reward_extra_infos_dict["verl_tool_metrics"].extend(
                    test_batch.non_tensor_batch.get("verl_tool_metrics", [None] * reward_tensor.shape[0])
                )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
        if "tool_interact_info" in reward_extra_infos_dict:
            # remove if after dump
            reward_extra_infos_dict.pop("tool_interact_info")
        if "traj_stop_reason" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("traj_stop_reason")
        if "verl_tool_metrics" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("verl_tool_metrics")

        # Structured turn rewards are consumed by MT-GRPO during training;
        # validation metrics expect scalar lists.
        reward_extra_infos_dict.pop("turn_reward", None)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        reward_mean_keys = [
            key
            for key, value in metric_dict.items()
            if key.startswith("val-core/")
            and key.endswith("/reward/mean@1")
            and isinstance(value, (int, float, np.number))
        ]
        if reward_mean_keys:
            metric_dict["val-core/all/reward/mean@1"] = float(
                np.mean([float(metric_dict[key]) for key in reward_mean_keys])
            )

        return metric_dict

    def _get_best_checkpoint_metric(self, val_metrics: dict):
        metric_key = self.config.trainer.get("save_best_metric", None)
        if metric_key:
            if metric_key not in val_metrics:
                available = ", ".join(sorted(val_metrics.keys()))
                raise KeyError(
                    f"trainer.save_best_metric={metric_key!r} is not in validation metrics. "
                    f"Available metrics: {available}"
                )
            return metric_key, float(val_metrics[metric_key])

        candidates = [
            key
            for key, value in val_metrics.items()
            if key.startswith("val-core/") and isinstance(value, (int, float, np.number))
        ]
        if not candidates:
            return None, None
        selected_key = sorted(candidates)[0]
        return selected_key, float(val_metrics[selected_key])

    def _is_better_checkpoint_metric(self, metric_value: float, best_metric_value: Optional[float]):
        if not math.isfinite(metric_value):
            return False
        if best_metric_value is None:
            return True

        mode = self.config.trainer.get("save_best_mode", "max")
        min_delta = float(self.config.trainer.get("save_best_min_delta", 0.0))
        if mode == "max":
            return metric_value > best_metric_value + min_delta
        if mode == "min":
            return metric_value < best_metric_value - min_delta
        raise ValueError(f"trainer.save_best_mode must be 'max' or 'min', got {mode!r}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs_attention_masks = batch.batch['attention_mask'][:, :batch.batch['prompts'].shape[1]]
            outputs_attention_masks = batch.batch['attention_mask'][:, batch.batch['prompts'].shape[1]:]
            inputs = [self.tokenizer.decode(batch.batch["prompts"][i][inputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["prompts"].shape[0])]
            outputs = [self.tokenizer.decode(batch.batch["responses"][i][outputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["responses"].shape[0])]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            
            tool_interact_info = batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                reward_extra_infos_to_dump.update({
                    "tool_interact_info": tool_interact_info,
                    "traj_stop_reason": batch.non_tensor_batch.get("traj_stop_reason", None),
                    "verl_tool_metrics": batch.non_tensor_batch.get("verl_tool_metrics", None),
                })

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
