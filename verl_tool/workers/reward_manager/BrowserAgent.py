import nltk
import json
import torch
import numpy as np

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
from verl_tool.utils.dense_reward import atomic_query_scores

from mini_webarena.rl_utils import format_score
from mini_webarena.evaluator import metric_heuristic
# ------------------------------------------------------------------------------
# WikiRL Reward Manager
# ------------------------------------------------------------------------------

OBS_ELEMENT_RE = re.compile(r"^[\t ]*(?:\[|<)(\d+)(?:\]|>)\s+([a-zA-Z]+)", re.MULTILINE)
ACTION_BLOCK_RE = re.compile(r"```(.*?)```|<action>(.*?)</action>", re.DOTALL)
TARGET_ACTION_RE = re.compile(r"^(click|type|hover|tab_focus)\s+(?:\[|<)(\d+)(?:\]|>)")


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _as_float(value, default):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _config_value(kwargs, key, env_key, default):
    return kwargs.get(key, os.getenv(env_key, default))


def extract_browser_action(action_text: str) -> str:
    action_text = (action_text or "").strip()
    matches = list(ACTION_BLOCK_RE.finditer(action_text))
    if matches:
        match = matches[-1]
        return (match.group(1) or match.group(2) or "").strip()
    return action_text


def obs_elements(observation: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2).lower()
        for match in OBS_ELEMENT_RE.finditer(observation or "")
    }


def is_element_mismatch(verb: str, target_type: str) -> bool:
    if verb == "type":
        return target_type not in {"textbox", "searchbox", "textarea", "combobox", "input"}
    if verb in {"click", "hover"}:
        return target_type in {
            "statictext",
            "heading",
            "rootwebarea",
            "row",
            "cell",
            "table",
            "group",
            "paragraph",
            "text",
        }
    return False


def browser_action_process_reward(tool_interact_info) -> dict[str, float]:
    """Score whether browser actions target valid, compatible elements."""
    if isinstance(tool_interact_info, np.ndarray):
        tool_interact_info = tool_interact_info.tolist()
    if not tool_interact_info:
        return {
            "action_correctness_score": 0.0,
            "process_penalty": 0.0,
            "hallucinated_id_penalty": 0.0,
            "tool_invalid_penalty": 0.0,
            "num_target_actions": 0,
            "num_correct_actions": 0,
            "num_hallucinated_ids": 0,
            "num_element_mismatches": 0,
            "num_tool_invalid": 0,
        }

    target_actions = 0
    correct_actions = 0
    incorrect_actions = 0
    hallucinated_ids = 0
    element_mismatches = 0
    tool_invalid = 0
    total_actions = 0
    previous_obs = ""

    for info in tool_interact_info:
        if not isinstance(info, dict):
            continue

        action = extract_browser_action(str(info.get("action", "")))
        if action:
            total_actions += 1
        if action and info.get("valid_action") in (0, False):
            tool_invalid += 1

        action_match = TARGET_ACTION_RE.match(action)
        if action_match:
            target_actions += 1
            target_exists = info.get("target_id_exists", None)
            type_match = info.get("element_type_match", None)
            action_is_tool_invalid = info.get("valid_action") in (0, False)

            if target_exists is None:
                elements = obs_elements(previous_obs)
                target_id = action_match.group(2)
                target_type = elements.get(target_id)
                target_exists = target_id in elements
                type_match = (
                    None
                    if target_type is None
                    else not is_element_mismatch(action_match.group(1), target_type)
                )

            if not target_exists:
                hallucinated_ids += 1
                incorrect_actions += 1
            elif type_match is False:
                element_mismatches += 1
                incorrect_actions += 1
            elif action_is_tool_invalid:
                incorrect_actions += 1
            else:
                correct_actions += 1

        obs = info.get("browser_obs_for_reward", info.get("obs", ""))
        if isinstance(obs, str):
            previous_obs = obs

    denom = target_actions if target_actions else 1
    action_correctness_score = (correct_actions - incorrect_actions) / denom
    hallucinated_id_penalty = -hallucinated_ids / denom
    tool_invalid_penalty = -tool_invalid / (total_actions if total_actions else 1)
    bad_actions = incorrect_actions + max(0, tool_invalid - incorrect_actions)
    process_penalty = -bad_actions / (total_actions if total_actions else 1)
    return {
        "action_correctness_score": float(action_correctness_score),
        "process_penalty": float(process_penalty),
        "hallucinated_id_penalty": float(hallucinated_id_penalty),
        "tool_invalid_penalty": float(tool_invalid_penalty),
        "num_target_actions": int(target_actions),
        "num_correct_actions": int(correct_actions),
        "num_hallucinated_ids": int(hallucinated_ids),
        "num_element_mismatches": int(element_mismatches),
        "num_tool_invalid": int(tool_invalid),
    }


def browser_dense_turn_rewards(
    tool_interact_info,
    retrieval_weight: float = 0.20,
    refinement_weight: float = 0.10,
    query_weight: float = 0.05,
    action_penalty_weight: float = 0.05,
    retrieval_decay: float = 0.05,
    repeat_retrieval_penalty_ratio: float = 0.25,
) -> list[dict]:
    """Build token-span aware turn rewards from compact browser interaction logs."""
    if isinstance(tool_interact_info, np.ndarray):
        tool_interact_info = tool_interact_info.tolist()

    infos = [info for info in (tool_interact_info or []) if isinstance(info, dict)]
    query_scores = atomic_query_scores(infos)
    turns: dict[int, dict] = {}
    repeat_retrieval_penalty_ratio = max(0.0, float(repeat_retrieval_penalty_ratio))

    def ensure_turn(turn) -> dict | None:
        if not isinstance(turn, (int, np.integer)) or int(turn) < 0:
            return None
        turn = int(turn)
        return turns.setdefault(
            turn,
            {
                "turn": turn,
                "response_start": -1,
                "response_end": -1,
                "retrieval": 0.0,
                "repeat_retrieval_penalty": 0.0,
                "refinement": 0.0,
                "query": 0.0,
                "action_penalty": 0.0,
                "reward": 0.0,
            },
        )

    for info in infos:
        action_turn_entry = ensure_turn(info.get("action_turn_index"))
        if action_turn_entry is not None:
            action = extract_browser_action(str(info.get("action", "")))
            decay = float(np.exp(-retrieval_decay * action_turn_entry["turn"]))
            retrieval_delta = float(info.get("retrieval_delta", 0.0) or 0.0)
            if retrieval_delta > 0:
                action_turn_entry["retrieval"] += retrieval_delta * decay
            elif (
                action
                and float(info.get("gt_match_score", 0.0) or 0.0) > 0.0
                and not bool(info.get("first_gt_hit", False))
            ):
                penalty = repeat_retrieval_penalty_ratio * decay
                action_turn_entry["retrieval"] -= penalty
                action_turn_entry["repeat_retrieval_penalty"] -= penalty
            action_turn_entry["query"] += float(query_scores.get(action_turn_entry["turn"], 0.0))

            bad_action = bool(action) and (
                info.get("valid_action") in (0, False)
                or info.get("target_id_exists") is False
                or info.get("element_type_match") is False
            )
            if bad_action:
                action_turn_entry["action_penalty"] = -1.0

        generation_turn_entry = ensure_turn(info.get("generation_turn_index"))
        if generation_turn_entry is not None:
            start = info.get("generation_response_start", -1)
            end = info.get("generation_response_end", -1)
            if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
                generation_turn_entry["response_start"] = int(start)
                generation_turn_entry["response_end"] = int(end)
            generation_turn_entry["refinement"] += float(info.get("refinement_delta", 0.0) or 0.0)

    result = []
    for turn in sorted(turns):
        entry = turns[turn]
        entry["reward"] = float(
            retrieval_weight * entry["retrieval"]
            + refinement_weight * entry["refinement"]
            + query_weight * entry["query"]
            + action_penalty_weight * entry["action_penalty"]
        )
        result.append(entry)
    return result

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
        self.fuzzy_weight = _as_float(
            _config_value(kwargs, "answer_weight", "BROWSER_AGENT_ANSWER_WEIGHT", 0.9),
            0.9,
        )
        self.structure_weight = _as_float(
            _config_value(kwargs, "format_weight", "BROWSER_AGENT_FORMAT_WEIGHT", 0.05),
            0.05,
        )
        self.enable_process_reward = _as_bool(
            _config_value(kwargs, "enable_process_reward", "BROWSER_AGENT_ENABLE_PROCESS_REWARD", True),
            True,
        )
        self.action_correctness_weight = _as_float(
            _config_value(
                kwargs,
                "action_correctness_weight",
                "BROWSER_AGENT_ACTION_CORRECTNESS_WEIGHT",
                0.0,
            ),
            0.0,
        )
        self.hallucinated_id_penalty_weight = _as_float(
            _config_value(
                kwargs,
                "hallucinated_id_penalty_weight",
                "BROWSER_AGENT_HALLUCINATED_ID_PENALTY_WEIGHT",
                0.0,
            ),
            0.0,
        )
        self.tool_invalid_penalty_weight = _as_float(
            _config_value(
                kwargs,
                "tool_invalid_penalty_weight",
                "BROWSER_AGENT_TOOL_INVALID_PENALTY_WEIGHT",
                0.0,
            ),
            0.0,
        )
        self.process_penalty_weight = _as_float(
            _config_value(
                kwargs,
                "process_penalty_weight",
                "BROWSER_AGENT_PROCESS_PENALTY_WEIGHT",
                0.05,
            ),
            0.05,
        )
        self.retrieval_reward_weight = _as_float(
            _config_value(
                kwargs,
                "retrieval_reward_weight",
                "BROWSER_AGENT_RETRIEVAL_REWARD_WEIGHT",
                0.20,
            ),
            0.20,
        )
        self.refinement_reward_weight = _as_float(
            _config_value(
                kwargs,
                "refinement_reward_weight",
                "BROWSER_AGENT_REFINEMENT_REWARD_WEIGHT",
                0.10,
            ),
            0.10,
        )
        self.query_reward_weight = _as_float(
            _config_value(
                kwargs,
                "query_reward_weight",
                "BROWSER_AGENT_QUERY_REWARD_WEIGHT",
                0.05,
            ),
            0.05,
        )
        self.action_penalty_turn_weight = _as_float(
            _config_value(
                kwargs,
                "action_penalty_turn_weight",
                "BROWSER_AGENT_ACTION_PENALTY_TURN_WEIGHT",
                0.05,
            ),
            0.05,
        )
        self.retrieval_decay = _as_float(
            _config_value(
                kwargs,
                "retrieval_decay",
                "BROWSER_AGENT_RETRIEVAL_DECAY",
                0.05,
            ),
            0.05,
        )
        self.repeat_retrieval_penalty_ratio = _as_float(
            _config_value(
                kwargs,
                "repeat_retrieval_penalty_ratio",
                "BROWSER_AGENT_REPEAT_RETRIEVAL_PENALTY_RATIO",
                0.0,
            ),
            0.25,
        )
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
        action_correctness_scores = []
        hallucinated_id_penalties = []
        tool_invalid_penalties = []
        process_penalties = []
        turn_rewards = []
        dense_reward_sums = []
        dense_retrieval_rewards = []
        dense_repeat_retrieval_penalties = []
        dense_refinement_rewards = []
        dense_query_rewards = []
        dense_action_penalties = []
        dense_num_turns = []
        num_target_actions = []
        num_correct_actions = []
        num_hallucinated_ids = []
        num_element_mismatches = []
        num_tool_invalid = []
        final_rewards = []
        tool_interact_batch = data.non_tensor_batch.get("tool_interact_info", [None] * len(data))

        for i in range(len(data)):
            gts = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            pred = response_list[i]
            answer_reward  = self.answer_score(pred, gts)
            uid = data.non_tensor_batch.get("uid", [None] * len(data))[i]
            format_reward = self.format_score(actions_list[i], uid=uid)
            tool_interact_info = (
                tool_interact_batch[i]
                if isinstance(tool_interact_batch, (list, tuple, np.ndarray)) and i < len(tool_interact_batch)
                else None
            )
            process_scores = browser_action_process_reward(tool_interact_info)
            action_correctness_reward = process_scores["action_correctness_score"] if self.enable_process_reward else 0.0
            hallucinated_id_penalty = process_scores["hallucinated_id_penalty"] if self.enable_process_reward else 0.0
            tool_invalid_penalty = process_scores["tool_invalid_penalty"] if self.enable_process_reward else 0.0
            process_penalty = process_scores["process_penalty"] if self.enable_process_reward else 0.0
            dense_turn_reward = (
                browser_dense_turn_rewards(
                    tool_interact_info,
                    retrieval_weight=self.retrieval_reward_weight,
                    refinement_weight=self.refinement_reward_weight,
                    query_weight=self.query_reward_weight,
                    action_penalty_weight=self.action_penalty_turn_weight,
                    retrieval_decay=self.retrieval_decay,
                    repeat_retrieval_penalty_ratio=self.repeat_retrieval_penalty_ratio,
                )
                if self.enable_process_reward
                else []
            )
            dense_reward_sum = float(sum(item.get("reward", 0.0) for item in dense_turn_reward))
            dense_retrieval_sum = float(sum(item.get("retrieval", 0.0) for item in dense_turn_reward))
            dense_repeat_retrieval_penalty_sum = float(
                sum(item.get("repeat_retrieval_penalty", 0.0) for item in dense_turn_reward)
            )
            dense_refinement_sum = float(sum(item.get("refinement", 0.0) for item in dense_turn_reward))
            dense_query_sum = float(sum(item.get("query", 0.0) for item in dense_turn_reward))
            dense_action_penalty_sum = float(sum(item.get("action_penalty", 0.0) for item in dense_turn_reward))
            final_reward = (
                self.fuzzy_weight * answer_reward +
                self.structure_weight * format_reward +
                self.process_penalty_weight * process_penalty +
                self.action_correctness_weight * action_correctness_reward +
                self.hallucinated_id_penalty_weight * hallucinated_id_penalty +
                self.tool_invalid_penalty_weight * tool_invalid_penalty
            )

            # reward_tensor[i, valid_resp_len[i].item() - 1] = final_reward
            # 将 final_reward 填入 sequence 的最后一个有效位置
            idx = max(0, int(valid_resp_len[i].item()) - 1)
            reward_tensor[i, idx] = final_reward

            answer_scores.append(answer_reward)
            format_scores.append(format_reward)
            action_correctness_scores.append(action_correctness_reward)
            hallucinated_id_penalties.append(hallucinated_id_penalty)
            tool_invalid_penalties.append(tool_invalid_penalty)
            process_penalties.append(process_penalty)
            turn_rewards.append({"turns": dense_turn_reward})
            dense_reward_sums.append(dense_reward_sum)
            dense_retrieval_rewards.append(dense_retrieval_sum)
            dense_repeat_retrieval_penalties.append(dense_repeat_retrieval_penalty_sum)
            dense_refinement_rewards.append(dense_refinement_sum)
            dense_query_rewards.append(dense_query_sum)
            dense_action_penalties.append(dense_action_penalty_sum)
            dense_num_turns.append(len(dense_turn_reward))
            num_target_actions.append(process_scores["num_target_actions"])
            num_correct_actions.append(process_scores["num_correct_actions"])
            num_hallucinated_ids.append(process_scores["num_hallucinated_ids"])
            num_element_mismatches.append(process_scores["num_element_mismatches"])
            num_tool_invalid.append(process_scores["num_tool_invalid"])
            final_rewards.append(final_reward)

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
        print("Action correctness scores:", action_correctness_scores)
        print("Hallucinated ID penalties:", hallucinated_id_penalties)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    # 把指标以 list 的形式传入，外层会提取 value[0] 记录到 wandb
                    "wiki_answer_score": answer_scores,
                    "wiki_format_score": format_scores,
                    "browser_action_correctness_score": action_correctness_scores,
                    "browser_hallucinated_id_penalty": hallucinated_id_penalties,
                    "browser_tool_invalid_penalty": tool_invalid_penalties,
                    "browser_process_penalty": process_penalties,
                    "browser_dense_reward_sum": dense_reward_sums,
                    "browser_dense_retrieval_reward": dense_retrieval_rewards,
                    "browser_dense_repeat_retrieval_penalty": dense_repeat_retrieval_penalties,
                    "browser_dense_refinement_reward": dense_refinement_rewards,
                    "browser_dense_query_reward": dense_query_rewards,
                    "browser_dense_action_penalty": dense_action_penalties,
                    "browser_turn_reward_num_turns": dense_num_turns,
                    "turn_reward": turn_rewards,
                    "browser_num_target_actions": num_target_actions,
                    "browser_num_correct_actions": num_correct_actions,
                    "browser_num_hallucinated_ids": num_hallucinated_ids,
                    "browser_num_element_mismatches": num_element_mismatches,
                    "browser_num_tool_invalid": num_tool_invalid,
                    "browser_final_reward": final_rewards,
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
