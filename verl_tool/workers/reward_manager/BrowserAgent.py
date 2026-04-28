import nltk
import json
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import os
import time
import asyncio
import regex as re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from verl.workers.reward_manager import register

from mini_webarena.rl_utils import format_score
from mini_webarena.evaluator import metric_heuristic
# ------------------------------------------------------------------------------
# WikiRL Reward Manager
# ------------------------------------------------------------------------------

def clean_text(text):
    # 删除控制字符 & 非打印字符
    return re.sub(r'[\x00-\x1F\x7F-\x9F\u200b-\u200f\u2028-\u202f\u2060-\u206f]', '', text)

@register("BrowserAgent")
class WikiRLRewardManager:
    """
    Reward Manager for the WikiRL dataset.

    This class computes a combined reward for each predicted answer by comparing it with
    the ground truth answers. The final reward is a weighted combination of a fuzzy matching
    score and a structure score.
    # """
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, **kwargs) -> None:
        """
        Initialize the WikiRLRewardManager.

        Parameters:
        - fuzzy_weight: The weight applied to the fuzzy matching score.
        - structure_weight: The weight applied to the structure score.
        """
        if tokenizer is None:
            # Simply use QWen2.5-7B tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.fuzzy_weight = 0.7
        self.structure_weight = 0.3
        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)

    def answer_score(self, pred, ground_truths):
        def extract_last_stop_content(input_str: str) -> str:
            matches = re.findall(r"```stop\s*\[([^\]]*)\]```", input_str)
            if matches:
                return matches[-1]
            return ""
        # First match ```stop [...]``` use regex to find the last ```stop [...]``` in the string
        pred = extract_last_stop_content(pred)
        score = metric_heuristic(ground_truths, pred)
        # print("answer score", ground_truths, pred, score)
        return score

    def format_score(self, actions, uid=None):
        scores = []

        for j, action in enumerate(actions):
            try:
                s = format_score(action)
            except Exception as e:
                s = 0.0
                print(f"[FORMAT_SCORE_ERROR] uid={uid} action_idx={j} err={repr(e)}")

            scores.append(s)

            # print("=" * 100)
            # print(f"[FORMAT_DEBUG] uid={uid} action_idx={j} score={s}")
            # print("[ACTION_REPR]")
            # print(repr(action))
            # print("[ACTION_RAW]")
            # print(action)
            # print("=" * 100)

        return sum(scores) / len(scores) if scores else 0.0

    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute scalar rewards for a batch and append per‑sample logs to
        ``reward_manager_history.jsonl``.

        Each JSON line now stores token‑separated strings (not raw ID lists):

        {
            "uid": <trajectory_uid>,
            "input_tokens": "▁This ▁is ▁a ...",      # whitespace‑joined tokens
            "pred_tokens": "▁Answer ▁text ...",
            "actions": [...],
            "observations": [...],
            "answer_score": <float>,
            "format_score": <float>
        }
        """
        # check the last step index
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        print("💰 wikiRL Reward Manager: computing rewards for a batch...")
        import pickle
        # with open("data_stub_new_qwq.pkl", "wb") as f:
        #     pickle.dump(data, f)

        import json
        from pathlib import Path

        special_token_ids = set(self.tokenizer.all_special_ids)

        actions_list, observations_list, response_list = [], [], []

        # ---------- 1. decode reward texts from responses --------------------
        for i in range(len(data)):
            uid = data.non_tensor_batch.get("uid", [None] * len(data))[i]

            # 1) 直接使用 responses，而不是 input_ids[prompt_len:]
            response_ids = data.batch["responses"][i].tolist()

            # 2) 保留 special tokens，方便识别 <|im_start|>assistant ... <|im_end|>
            decoded_response_with_special = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=False
            )

            decoded_response_no_special = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True
            ).strip()

            # 3) 提取所有 assistant block
            assistant_blocks = re.findall(
                r"<\|im_start\|>assistant\s*(.*?)(?=<\|im_end\|>)",
                decoded_response_with_special,
                flags=re.DOTALL
            )

            # 4) 如果最后一个 assistant 没有 <|im_end|>，补充提取尾部残段
            if "<|im_start|>assistant" in decoded_response_with_special:
                last_after_assistant = decoded_response_with_special.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" not in last_after_assistant:
                    tail_block = last_after_assistant.strip()
                    if tail_block:
                        assistant_blocks.append(tail_block)

            # 5) 清理 assistant block 里的 special tokens
            cleaned_assistant_blocks = []
            for block in assistant_blocks:
                block = re.sub(r"<\|.*?\|>", "", block)
                block = block.strip()
                if block:
                    cleaned_assistant_blocks.append(block)

            # 6) 用于 answer_score 的完整轨迹文本：所有 assistant 输出拼接
            if cleaned_assistant_blocks:
                reward_text = "\n\n".join(cleaned_assistant_blocks)
            else:
                # fallback：如果没匹配到 assistant block，就用 skip_special_tokens 后的 responses
                reward_text = decoded_response_no_special

            response_list.append(reward_text)

            # 7) 用于 format_score 的输入：
            #    不改奖励计算逻辑，仍然把 actions_list[i] 传给 self.format_score()
            #    这里把每个 assistant block 作为一个 action 评分单元
            actions = cleaned_assistant_blocks if cleaned_assistant_blocks else [reward_text]
            observations = []

            actions_list.append(actions)
            observations_list.append(observations)

            # 8) 记录一条轨迹所有用于奖励计算的文本
            try:
                format_reward_text = "\n\n".join(
                    f"===== ACTION {j} =====\n{a}"
                    for j, a in enumerate(actions)
                )

                debug_entry = {
                    "uid": str(uid),
                    "sample_idx": i,
                    "num_actions": len(actions),

                    # 最关键：实际用于 format_reward 的所有文本
                    "format_reward_text": format_reward_text,
                    "format_reward_actions": actions,
                    "format_reward_actions_repr": [repr(a) for a in actions],

                    # 实际用于 answer_score 的文本
                    "reward_text": reward_text,
                    "reward_text_repr": repr(reward_text),

                    # 原始 responses 解码，方便排查
                    # "decoded_response_with_special": decoded_response_with_special,
                    # "decoded_response_with_special_repr": repr(decoded_response_with_special),
                    # "decoded_response_no_special": decoded_response_no_special,
                    # "decoded_response_no_special_repr": repr(decoded_response_no_special),
                }

                # with Path("format_reward_input_debug.jsonl").open("a", encoding="utf-8") as f:
                #     f.write(json.dumps(debug_entry, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[WARN] could not write format_reward_input_debug.jsonl: {e}")

        # ---------- 2.  reward tensor --------------------------------------
        prompt_ids   = data.batch["prompts"]
        prompt_len   = prompt_ids.shape[-1]
        responses_id = data.batch["responses"]
        valid_resp_len = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
        reward_tensor = torch.zeros_like(responses_id, dtype=torch.float32)

        answer_scores, format_scores = [], []

        for i in range(len(data)):
            gts = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            pred = response_list[i]
            answer_reward  = self.answer_score(pred, gts)
            uid = data.non_tensor_batch.get("uid", [None] * len(data))[i]
            format_reward = self.format_score(actions_list[i], uid=uid)
            final_reward = (
                self.fuzzy_weight * answer_reward +
                self.structure_weight * format_reward
            )

            # reward_tensor[i, valid_resp_len[i].item() - 1] = final_reward
            # 将 final_reward 填入 sequence 的最后一个有效位置
            idx = max(0, int(valid_resp_len[i].item()) - 1)
            reward_tensor[i, idx] = final_reward

            answer_scores.append(answer_reward)
            format_scores.append(format_reward)

        # ---------- 3.  persistent logging ---------------------------------
        # try:
        #     log_file = Path("/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/logs/reward_manager_history.jsonl")
        #     log_file.parent.mkdir(parents=True, exist_ok=True)
        #     with log_file.open("a", encoding="utf-8") as f:
        #         for idx in range(len(data)):
        #             # convert entire sequence and prediction to whitespace‑joined tokens
        #             input_text = clean_text(self.tokenizer.decode(
        #                 data.batch["input_ids"][idx].tolist(),
        #                 skip_special_tokens=True
        #             ).strip())
        #             input_tokens = " ".join(self.tokenizer.tokenize(input_text))
        #             pred_tokens = " ".join(self.tokenizer.tokenize(clean_text(response_list[idx])))

        #             log_entry = {
        #                 "uid": data.non_tensor_batch.get("uid", [None]*len(data))[idx],
        #                 "input_tokens": input_tokens,
        #                 "pred_tokens": pred_tokens,

        #                 # 原有字段
        #                 "actions": actions_list[idx],
        #                 "observations": observations_list[idx],

        #                 # ==================== 新增：实际传入 format_score 的全部文本 ====================
        #                 "format_reward_text": "\n\n".join(
        #                     f"===== ACTION {j} =====\n{a}"
        #                     for j, a in enumerate(actions_list[idx])
        #                 ),
        #                 "format_reward_actions_repr": [
        #                     repr(a) for a in actions_list[idx]
        #                 ],
        #                 # =====================================================================

        #                 "answer_score": answer_scores[idx],
        #                 "format_score": format_scores[idx],
        #             }
        #             f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        # except Exception as e:
        #     print(f"[WARN] could not append to reward_manager_history.jsonl: {e}")

        print(f"Computed rewards for {len(data)} samples.")
        print("Answer scores:", answer_scores)
        print("Format scores:", format_scores)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    # 把指标以 list 的形式传入，外层会提取 value[0] 记录到 wandb
                    "wiki_answer_score": answer_scores,
                    "wiki_format_score": format_scores,
                }
            }
        
        return reward_tensor


if __name__ == '__main__':
    import pickle

    # Load the saved data object from disk
    with open("data_stub_new.pkl", "rb") as f:
        dummy_data = pickle.load(f)

    # Instantiate the WikiRLRewardManager (you can pass in config if needed)
    reward_manager = WikiRLRewardManager()

    # Compute rewards for the loaded data
    rewards = reward_manager(dummy_data)
    print("Rewards:", rewards)


"""
(TaskRunner pid=2019847) ==== Call WikiRLRewardManager ====
(TaskRunner pid=2019847) DataProto(batch=TensorDict(
(TaskRunner pid=2019847)     fields={
(TaskRunner pid=2019847)         attention_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         loss_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         input_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         old_log_probs: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         position_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         prompts: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         ref_log_prob: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         responses: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         responses_with_loss_mask: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False)},
(TaskRunner pid=2019847)     batch_size=torch.Size([4]),
(TaskRunner pid=2019847)     device=None,
(TaskRunner pid=2019847)     is_shared=False), non_tensor_batch={'data_source': array(['wiki_qa', 'wiki_qa', 'wiki_qa', 'wiki_qa'], dtype=object), 'ability': array(['wiki', 'wiki', 'wiki', 'wiki'], dtype=object), 'reward_model': array([{'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'}],
(TaskRunner pid=2019847)       dtype=object), 'index': array([0, 0, 0, 0], dtype=object), 'uid': array(['ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b'], dtype=object)}, meta_info={'turns_stats': [4, 4], 'active_mask': [True, True], 'valid_action_stats': [4, 4], 'global_token_num': [5541, 5541, 3697, 5542], 'temperature': 0.9})
"""
