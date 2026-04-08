import ray
import uuid
import torch
import os
import json
import numpy as np
from copy import deepcopy
from pprint import pprint
from collections import defaultdict
from typing import Optional
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    AdvantageEstimator,
    Role,
    agg_loss,
    align_batch_to_rollout_output,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    compute_throughout_metrics,
    compute_timing_metrics,
    pad_dataproto_to_divisor,
    reduce_metrics,
    RolloutSkip,
    should_save_ckpt_esi,
    unpad_dataproto,
    process_validation_metrics,
) # for train and validate
from verl.trainer.ppo.ray_trainer import (
    DataProto,
) # for init
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from tqdm import tqdm
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


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
    def _is_himac_training_enabled(self) -> bool:
        return bool(self.config.actor_rollout_ref.agent.get("enable_blueprint_rollout", False))

    def _get_phase_repeat_times(self, phase: str) -> int:
        key = "macro_group_size" if phase == "macro" else "micro_group_size"
        repeat_times = self.config.actor_rollout_ref.agent.get(key, self.config.actor_rollout_ref.rollout.n)
        return max(int(repeat_times), 1)

    def _prefix_metrics(self, prefix: str, metrics: dict[str, float]) -> dict[str, float]:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}

    def _set_rollout_phase(
        self,
        gen_batch: DataProto,
        *,
        phase: str,
        fixed_blueprints: Optional[list[str]] = None,
    ) -> DataProto:
        gen_batch.non_tensor_batch["rollout_phase"] = np.array([phase] * len(gen_batch), dtype=object)
        if fixed_blueprints is not None:
            gen_batch.non_tensor_batch["fixed_blueprint"] = np.array(fixed_blueprints, dtype=object)
        return gen_batch

    def _rollout_with_phase(
        self,
        source_batch: DataProto,
        gen_batch: DataProto,
        *,
        phase: str,
        repeat_times: int,
        timing_raw: dict,
    ) -> DataProto:
        phase_gen_batch = deepcopy(gen_batch)
        phase_gen_batch = self._set_rollout_phase(phase_gen_batch, phase=phase)
        phase_gen_batch.meta_info["global_steps"] = self.global_steps
        phase_gen_batch_output = phase_gen_batch.repeat(repeat_times=repeat_times, interleave=True)

        with marked_timer(f"{phase}_gen", timing_raw, color="red"):
            if not self.async_rollout_mode:
                phase_gen_batch_output = self.actor_rollout_wg.generate_sequences(phase_gen_batch_output)
            else:
                phase_gen_batch_output = self.async_rollout_manager.generate_sequences(phase_gen_batch_output)

        phase_timing = phase_gen_batch_output.meta_info.pop("timing", {})
        timing_raw.update({f"{phase}/{key}": value for key, value in phase_timing.items()})

        phase_batch = source_batch.repeat(repeat_times=repeat_times, interleave=True)
        phase_batch = align_batch_to_rollout_output(phase_batch, phase_gen_batch_output)
        phase_batch = phase_batch.union(phase_gen_batch_output)
        return phase_batch

    def _prepare_phase_batch(
        self,
        batch: DataProto,
        *,
        phase: str,
        metrics: dict[str, float],
        timing_raw: dict,
    ) -> tuple[DataProto, dict[str, list]]:
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics, logging_prefix=f"{phase}_seqlen")

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        with marked_timer(f"{phase}_reward", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer)
                reward_tensor = None
                reward_extra_infos_dict = {}
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        with marked_timer(f"{phase}_old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                from verl.utils.debug.metrics import calculate_debug_metrics

                metrics.update(calculate_debug_metrics(batch))

        if self.use_reference_policy:
            with marked_timer(f"{phase}_{str(Role.RefPolicy)}", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer(f"{phase}_values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer(f"{phase}_adv", timing_raw, color="brown"):
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
            metrics.update(is_metrics)

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=1,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        return batch, reward_extra_infos_dict

    def _select_best_blueprints(self, macro_batch: DataProto, source_uids: np.ndarray) -> tuple[list[str], list[float]]:
        scores = macro_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().tolist()
        macro_uids = macro_batch.non_tensor_batch["uid"]
        blueprint_texts = macro_batch.non_tensor_batch.get("blueprint_text")
        if blueprint_texts is None:
            raise ValueError("Macro rollout output must include `blueprint_text` for HiMAC micro stage.")

        best_blueprints: list[str] = []
        best_scores: list[float] = []
        for uid in source_uids.tolist():
            candidate_indices = [idx for idx, sample_uid in enumerate(macro_uids.tolist()) if sample_uid == uid]
            if not candidate_indices:
                raise ValueError(f"Cannot find blueprint candidates for uid={uid}")
            best_idx = max(candidate_indices, key=lambda idx: scores[idx])
            best_blueprints.append(str(blueprint_texts[best_idx]))
            best_scores.append(float(scores[best_idx]))
        return best_blueprints, best_scores

    def _run_micro_phase(
        self,
        source_batch: DataProto,
        gen_batch: DataProto,
        *,
        best_blueprints: list[str],
        repeat_times: int,
        timing_raw: dict,
    ) -> DataProto:
        phase_gen_batch = deepcopy(gen_batch)
        phase_gen_batch = self._set_rollout_phase(
            phase_gen_batch,
            phase="micro",
            fixed_blueprints=best_blueprints,
        )
        phase_gen_batch.meta_info["global_steps"] = self.global_steps
        phase_gen_batch_output = phase_gen_batch.repeat(repeat_times=repeat_times, interleave=True)

        with marked_timer("micro_gen", timing_raw, color="red"):
            if not self.async_rollout_mode:
                phase_gen_batch_output = self.actor_rollout_wg.generate_sequences(phase_gen_batch_output)
            else:
                phase_gen_batch_output = self.async_rollout_manager.generate_sequences(phase_gen_batch_output)

        phase_timing = phase_gen_batch_output.meta_info.pop("timing", {})
        timing_raw.update({f"micro/{key}": value for key, value in phase_timing.items()})

        phase_batch = source_batch.repeat(repeat_times=repeat_times, interleave=True)
        phase_batch = align_batch_to_rollout_output(phase_batch, phase_gen_batch_output)
        phase_batch = phase_batch.union(phase_gen_batch_output)
        return phase_batch

    def _update_phase_models(self, batch: DataProto, *, phase: str, metrics: dict[str, float], timing_raw: dict) -> None:
        if self.use_critic:
            with marked_timer(f"{phase}_update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            metrics.update(self._prefix_metrics(phase, reduce_metrics(critic_output.meta_info["metrics"])))

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer(f"{phase}_update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            metrics.update(self._prefix_metrics(phase, reduce_metrics(actor_output.meta_info["metrics"])))

    
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

            if self._is_himac_training_enabled():
                val_phase = self.config.actor_rollout_ref.agent.get("validation_phase", "micro")
                test_batch.non_tensor_batch["rollout_phase"] = np.array([val_phase] * len(test_batch), dtype=object)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

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

            test_batch = align_batch_to_rollout_output(test_batch, test_output_gen_batch)

            # Store original inputs after rollout expansion so logging and metrics stay aligned with chunked outputs.
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

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

        return metric_dict

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

    def fit(self):
        if not self._is_himac_training_enabled():
            return super().fit()

        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                source_uids = batch.non_tensor_batch["uid"].copy()

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                macro_repeat_times = self._get_phase_repeat_times("macro")
                micro_repeat_times = self._get_phase_repeat_times("micro")
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    macro_batch = self._rollout_with_phase(
                        batch,
                        gen_batch,
                        phase="macro",
                        repeat_times=macro_repeat_times,
                        timing_raw=timing_raw,
                    )
                    macro_metrics = {}
                    macro_batch, macro_reward_extra_infos = self._prepare_phase_batch(
                        macro_batch,
                        phase="macro",
                        metrics=macro_metrics,
                        timing_raw=timing_raw,
                    )
                    metrics.update(self._prefix_metrics("macro", macro_metrics))

                    best_blueprints, best_scores = self._select_best_blueprints(macro_batch, source_uids)
                    metrics["himac/macro/best_blueprint_reward_mean"] = float(np.mean(best_scores)) if best_scores else 0.0
                    metrics["himac/macro/best_blueprint_reward_max"] = float(np.max(best_scores)) if best_scores else 0.0

                    micro_batch = self._run_micro_phase(
                        batch,
                        gen_batch,
                        best_blueprints=best_blueprints,
                        repeat_times=micro_repeat_times,
                        timing_raw=timing_raw,
                    )
                    micro_metrics = {}
                    micro_batch, micro_reward_extra_infos = self._prepare_phase_batch(
                        micro_batch,
                        phase="micro",
                        metrics=micro_metrics,
                        timing_raw=timing_raw,
                    )
                    metrics.update(self._prefix_metrics("micro", micro_metrics))

                    self._update_phase_models(macro_batch, phase="macro", metrics=metrics, timing_raw=timing_raw)
                    self._update_phase_models(micro_batch, phase="micro", metrics=metrics, timing_raw=timing_raw)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(macro_batch, macro_reward_extra_infos, timing_raw, rollout_data_dir)
                        self._log_rollout_data(micro_batch, micro_reward_extra_infos, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
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
                    }
                )
                metrics.update(self._prefix_metrics("macro", compute_data_metrics(batch=macro_batch, use_critic=self.use_critic)))
                metrics.update(self._prefix_metrics("micro", compute_data_metrics(batch=micro_batch, use_critic=self.use_critic)))
                metrics.update(compute_timing_metrics(batch=micro_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=micro_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=micro_batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

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
                    self.train_dataset.on_batch_end(batch=micro_batch)
