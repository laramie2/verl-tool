# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
import aiohttp
import asyncio
import numpy as np
import json
import uuid
import threading
import regex as re
from PIL import Image
from typing import Union, Optional, Dict
from typing import Any
from uuid import uuid4
from verl.utils.profiler import simple_timer
from .agent_loop import AgentLoopBase, register, AgentLoopOutput
from .vision_utils import decode_image_url, encode_image, encode_image_url
from verl_tool.utils.dataset.audio_utils import encode_audio_data
import socket

import textwrap
from PIL import Image, ImageDraw, ImageFont

from VTC_tool.VTC_tool import VTCTool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
from datetime import datetime
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

from dataclasses import dataclass

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)
MAX_OBS_LENGTH = 100000  # maximum observation length to send back to the tool server

async def on_connection_queued_start(session, trace_config_ctx, params):
    if hasattr(trace_config_ctx, 'trace_request_ctx') and trace_config_ctx.trace_request_ctx is not None:
        trace_config_ctx.trace_request_ctx['queued_start'] = asyncio.get_event_loop().time()

async def on_connection_queued_end(session, trace_config_ctx, params):
    if hasattr(trace_config_ctx, 'trace_request_ctx') and trace_config_ctx.trace_request_ctx is not None:
        if 'queued_start' in trace_config_ctx.trace_request_ctx:
            trace_config_ctx.trace_request_ctx['queued_duration'] = asyncio.get_event_loop().time() - trace_config_ctx.trace_request_ctx['queued_start']

async def on_connection_create_start(session, trace_config_ctx, params):
    if hasattr(trace_config_ctx, 'trace_request_ctx') and trace_config_ctx.trace_request_ctx is not None:
        trace_config_ctx.trace_request_ctx['create_start'] = asyncio.get_event_loop().time()

async def on_connection_create_end(session, trace_config_ctx, params):
    if hasattr(trace_config_ctx, 'trace_request_ctx') and trace_config_ctx.trace_request_ctx is not None:
        if 'create_start' in trace_config_ctx.trace_request_ctx:
            trace_config_ctx.trace_request_ctx['create_duration'] = asyncio.get_event_loop().time() - trace_config_ctx.trace_request_ctx['create_start']

async def on_request_start(session, trace_config_ctx, params):
    if hasattr(trace_config_ctx, 'trace_request_ctx') and trace_config_ctx.trace_request_ctx is not None:
        trace_config_ctx.trace_request_ctx['request_start'] = asyncio.get_event_loop().time()

async def on_request_end(session, trace_config_ctx, params):
    if hasattr(trace_config_ctx, 'trace_request_ctx') and trace_config_ctx.trace_request_ctx is not None:
        if 'request_start' in trace_config_ctx.trace_request_ctx:
            trace_config_ctx.trace_request_ctx['request_duration'] = asyncio.get_event_loop().time() - trace_config_ctx.trace_request_ctx['request_start']

@dataclass
class AgentActorConfig:
    enable_agent: bool=True
    max_turns: int=0
    val_max_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    train_max_response_length: int=None
    val_max_response_length: int=None
    max_model_len: int=None  # Maximum model length, used for async rollout to limit the input length.
    max_obs_length: int=None
    max_action_length: int=None
    tool_server_url: str = None
    n: int=1
    truncate_obs_side: str='middle'
    truncate_response_side: str='left'
    action_stop_tokens: list=None
    action_extract_tokens: list=None
    mask_observations: bool=True
    force_finish_for_last_turn: bool=False
    enable_mtrl: bool=False
    mtrl_role: str="user"
    mtrl_sep: str=None # "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    assistant_role: str="assistant"
    turn_end_token: str="<|im_end|>"
    rollout_mode: str="async" # "sync" or "async"
    mask_overlong_loss: bool=False # whether to mask the overlong trajectory to not train on it
    enable_tqdm: bool=True # Whether to enable tqdm for async rollout.
    tool_call_timeout: int=None # Timeout for tool calls in async rollout.
    tool_call_max_retries: int=1 # Maximum number of retries for tool calls in async rollout.
    max_concurrent_trajectories: int=1024 # Maximum number of concurrent trajectories globally for async rollout for all agent workers. If None, no limit is applied.
    
    # The following fields are to be disgarded, only for compatibility
    rolling_with_prompt: bool=False
    call_tool_first: bool=False
    additional_eos_token_ids: list=None
    over_sampling: bool=False # Whether to over-sample the trajectories in async rollout.
    min_turns: int=0
    retokenization: bool=False
    
    # experimental features
    mask_void_traj: bool=False # whether to mask the void trajectory (no tool call and no final answer)
    logprobs: bool=False # whether to return logprobs for each generated token
    compact_tool_interact_info: bool=False # whether to keep only lightweight tool interaction info in training batches
    
def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    elif isinstance(obj,Image.Image):
        return encode_image(obj)
    else:
        return obj


def compact_tool_interact_info_entries(tool_interact_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop large observation payloads while keeping MT-GRPO-relevant signals."""
    compacted = []
    for info in tool_interact_info:
        if info is None:
            compacted.append(None)
            continue

        obs = info.get("obs", "")
        obs_is_str = isinstance(obs, str)
        obs_nonempty = bool(obs.strip()) if obs_is_str else False
        obs_lower = obs.lower() if obs_is_str else ""

        compacted.append(
            {
                "valid_action": bool(info.get("valid_action", False)),
                "success": info.get("success", None),
                "done": bool(info.get("done", False)),
                "finish": bool(info.get("finish", False)),
                "reward": info.get("reward", None),
                "invalid_reason": info.get("invalid_reason", None),
                "obs_nonempty": obs_nonempty,
                "obs_is_error": (
                    ("Error:" in obs)
                    or ("Traceback" in obs)
                    or ("exception" in obs_lower)
                )
                if obs_is_str
                else False,
                "obs_has_result_or_output": (
                    ("<result>" in obs) or ("<output>" in obs)
                )
                if obs_is_str
                else False,
            }
        )
    return compacted
    
@register("verltool_agent")
class VerlToolAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""
    _init_lock = threading.Lock()
    _session: aiohttp.ClientSession | None = None
    _session_lock = asyncio.Lock()
    _rollout_log_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
    
        
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        with cls._init_lock:
            cls.tokenizer = tokenizer
            cls.processor = processor
            
            agent_config = AgentActorConfig()
            rollout_config = config.actor_rollout_ref
            for key in getattr(rollout_config, 'agent', {}).keys():
                if key in agent_config.__dict__.keys():
                    setattr(agent_config, key, rollout_config.agent[key])
            setattr(agent_config, 'n', rollout_config.rollout.n)
            setattr(agent_config, 'max_model_len', rollout_config.rollout.max_model_len)
            setattr(agent_config, 'rollout_mode', "async")
            cls.agent_config = agent_config

            if cls.agent_config.action_stop_tokens is not None:
                if os.path.exists(cls.agent_config.action_stop_tokens):
                    with open(cls.agent_config.action_stop_tokens, 'r') as f:
                        cls.action_stop_tokens = [x.strip() for x in f.read().split(',') if x.strip()]
                    logger.info(f"Using action stop tokens: {cls.action_stop_tokens}")
                else:
                    raise ValueError(f"action_stop_tokens file not found: {cls.agent_config.action_stop_tokens}")
            else:
                cls.action_stop_tokens = []

            # 新增 action_extract_tokens
            if getattr(cls.agent_config, 'action_extract_tokens', None) is not None:
                if os.path.exists(cls.agent_config.action_extract_tokens):
                    with open(cls.agent_config.action_extract_tokens, 'r') as f:
                        cls.action_extract_tokens = [x.strip() for x in f.read().split(',') if x.strip()]
                    logger.info(f"Using action extract tokens: {cls.action_extract_tokens}")
                else:
                    raise ValueError(f"action_extract_tokens file not found: {cls.agent_config.action_extract_tokens}")
            else:
                # 兼容老版本：如果没传，默认等于 stop_tokens
                cls.action_extract_tokens = getattr(cls, 'action_stop_tokens', [])

            if cls.agent_config.mtrl_sep is None:
                messages = [{"role": "system", "content": "{obs}"}]
                cls.agent_config.mtrl_sep = cls.agent_config.turn_end_token + "\n" + cls.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                cls.agent_config.mtrl_sep = cls.agent_config.mtrl_sep.replace("system", cls.agent_config.mtrl_role)
                cls.mtrl_sep_prefix, cls.mtrl_sep_suffix = cls.agent_config.mtrl_sep.split("{obs}")
                cls.mtrl_sep_prefix_ids = cls.tokenizer.encode(cls.mtrl_sep_prefix)
                cls.mtrl_sep_suffix_ids = cls.tokenizer.encode(cls.mtrl_sep_suffix)
            cls.enable_mtrl = cls.agent_config.enable_mtrl
            cls.mtrl_sep = cls.agent_config.mtrl_sep
            cls.max_model_len = int(cls.agent_config.max_model_len or (cls.agent_config.max_prompt_length + cls.agent_config.max_response_length))
            
            cls.response_length = rollout_config.rollout.response_length
            cls.max_response_length = cls.agent_config.max_response_length if cls.agent_config.max_response_length is not None else cls.response_length
            cls.max_response_length = min(cls.max_response_length, cls.max_model_len)
            cls.val_max_response_length = cls.agent_config.val_max_response_length if cls.agent_config.val_max_response_length is not None else cls.max_response_length
            cls.train_max_response_length = cls.agent_config.train_max_response_length if cls.agent_config.train_max_response_length is not None else cls.max_response_length
            assert cls.response_length >= cls.max_response_length, f"rollout.response_length {cls.response_length} must be >= agent.max_response_length {cls.max_response_length}"
            assert cls.response_length >= cls.val_max_response_length, f"rollout.response_length {cls.response_length} must be >= agent.val_max_response_length {cls.val_max_response_length}"
            assert cls.response_length >= cls.train_max_response_length, f"rollout.response_length {cls.response_length} must be >= agent.train_max_response_length {cls.train_max_response_length}"
            cls.max_action_length = cls.agent_config.max_action_length
            cls.max_obs_length = cls.agent_config.max_obs_length if cls.agent_config.max_obs_length is not None else 512
            cls.max_turns = cls.agent_config.max_turns if cls.agent_config.max_turns > 0 else 10
            cls.val_max_turns = cls.agent_config.val_max_turns if cls.agent_config.val_max_turns > 0 else cls.agent_config.max_turns
            cls.logprobs = cls.agent_config.logprobs
            cls.session_limit = cls.agent_config.max_concurrent_trajectories
            
            # for multimodal processing
            cls.qwen_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            cls.qwen_video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
            cls.qwen_audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"  # for qwen2.5-omni
            cls.non_truncate_tokens = [
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
                "<|video_pad|>",
                "<|audio_bos|>",
                "<|AUDIO|>",
                "<|audio_eos|>",
            ]
            cls.non_truncate_token_ids = [cls.tokenizer.convert_tokens_to_ids(tok) for tok in cls.non_truncate_tokens]
            
            
            if cls._class_initialized:
                return
            cls._class_initialized = True

            cls.vtc = VTCTool()
        
    @classmethod
    async def get_session(cls, timeout_seconds: float) -> aiohttp.ClientSession:
        # Double-checked locking for lazy init
        if cls._session is None or cls._session.closed:
            async with cls._session_lock:
                if cls._session is None or cls._session.closed:
                    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                    
                    trace_config = aiohttp.TraceConfig()
                    trace_config.on_connection_queued_start.append(on_connection_queued_start)
                    trace_config.on_connection_queued_end.append(on_connection_queued_end)
                    trace_config.on_connection_create_start.append(on_connection_create_start)
                    trace_config.on_connection_create_end.append(on_connection_create_end)
                    trace_config.on_request_start.append(on_request_start)
                    trace_config.on_request_end.append(on_request_end)
                    
                    cls._session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=aiohttp.TCPConnector(
                            limit=cls.session_limit,
                            limit_per_host=cls.session_limit,
                            ttl_dns_cache=300,
                            family=socket.AF_INET,
                        ),
                        trace_configs=[trace_config]
                    )
                    logging.info(f"Created shared aiohttp ClientSession for AgentLoopWorker with limit={cls.session_limit}")
        return cls._session

    @classmethod
    async def aclose_session(cls):
        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
            cls._session = None
            logging.info("Closed shared aiohttp ClientSession for AgentLoopWorker")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [VerlToolAgentLoop._json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(VerlToolAgentLoop._json_safe(k)): VerlToolAgentLoop._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [VerlToolAgentLoop._json_safe(item) for item in value]
        if isinstance(value, Image.Image):
            return f"<PIL.Image mode={value.mode} size={value.size}>"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _rollout_log_epoch(config, global_step: int, epoch: Any, validate: bool) -> Any:
        if epoch is not None:
            try:
                return int(epoch)
            except (TypeError, ValueError):
                return str(epoch)

        if global_step is None or global_step < 0:
            return "unknown"

        total_epochs = int(config.trainer.get("total_epochs", 0) or 0)
        total_steps = config.actor_rollout_ref.actor.optim.get("total_training_steps", None)
        if total_steps in (None, -1):
            total_steps = config.trainer.get("total_training_steps", None)
        if total_epochs <= 0 or total_steps in (None, -1):
            return "unknown"

        steps_per_epoch = max(int(total_steps) // total_epochs, 1)
        return int(global_step) // steps_per_epoch

    @staticmethod
    def _rollout_log_path(log_dir: str, epoch: Any, validate: bool) -> str:
        if isinstance(epoch, int):
            epoch_name = f"{epoch:04d}"
        else:
            epoch_name = str(epoch)
        prefix = "validation_epoch" if validate else "epoch"
        return os.path.join(log_dir, f"{prefix}_{epoch_name}.jsonl")

    @staticmethod
    def _rollout_log_dir(config) -> str:
        trainer_config = config.get("trainer", {})
        project_name = str(trainer_config.get("project_name", "unknown_project"))
        experiment_name = str(trainer_config.get("experiment_name", "unknown_experiment"))

        def safe_name(name: str) -> str:
            name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("._")
            return name or "unknown"

        return str(
            project_root
            / "RL"
            / "logs"
            / "verltool_agent_loop"
            / safe_name(project_name)
            / safe_name(experiment_name)
        )

    @classmethod
    def _append_rollout_jsonl(cls, log_filename: str, context: dict[str, Any], event: str, payload: dict[str, Any]):
        record = {
            "schema_version": 1,
            "written_at": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **context,
            **payload,
        }
        line = json.dumps(cls._json_safe(record), ensure_ascii=False) + "\n"

        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        with cls._rollout_log_lock:
            with open(log_filename, "a", encoding="utf-8") as f:
                try:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(line)
                    f.flush()
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    f.write(line)
                    f.flush()

    async def _aiohttp_request(self, payload: dict):
        max_retries = self.agent_config.tool_call_max_retries
        timeout_seconds = self.agent_config.tool_call_timeout

        session_request_time = 0
        session = await self.get_session(timeout_seconds)
        
        start_time = time.time()
        for attempt in range(max_retries + 1):
            try:
                trace_ctx = {}

                # === [新增打印] 观察发给 Tool Server 的数据 ===
                # print(f"\n[HTTP POST Payload to Tool Server]:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n", flush=True)
                # ============================================

                async with session.post(
                    url=self.agent_config.tool_server_url,
                    json=payload,
                    trace_request_ctx=trace_ctx
                ) as resp:
                    resp_start_time = time.time()
                    resp.raise_for_status()
                    
                    # Measure response read time and size
                    read_start = time.time()
                    resp_data = await resp.json()
                    read_duration = time.time() - read_start

                    # ================= [新增打印 2：工具返回的响应] =================
                    # print(f"\n[HTTP Response from Tool Server]:\n{json.dumps(resp_data, indent=2, ensure_ascii=False)}\n", flush=True)
                    # ==============================================================
                    
                    # Calculate approximate response size
                    try:
                        resp_size = len(json.dumps(resp_data))
                    except:
                        resp_size = 0
                        
                    resp_end_time = time.time()
                    resp_data['time awaiting response_ms'] = (resp_end_time - resp_start_time) * 1000.0
                    resp_data['success'] = True
                    resp_data['client_queued_ms'] = trace_ctx.get('queued_duration', 0.0) * 1000.0
                    resp_data['client_conn_create_ms'] = trace_ctx.get('create_duration', 0.0) * 1000.0
                    resp_data['client_request_send_ms'] = trace_ctx.get('request_duration', 0.0) * 1000.0
                    resp_data['request_start_ms'] = trace_ctx.get('request_start', 0.0) * 1000.0
                    resp_data['response_read_ms'] = read_duration * 1000.0
                    resp_data['response_size_mb'] = resp_size / (1024 * 1024)
                    resp_data['X-Process-Time'] = float(resp.headers.get('X-Process-Time', '0.0'))
                    
                    if resp_data['response_size_mb'] > 1.0:
                        logging.warning(f"Large response received: {resp_data['response_size_mb']:.2f} MB. Read time: {read_duration:.2f}s")
                    break

            except asyncio.TimeoutError as e:
                logging.error(
                    f"Attempt {attempt + 1} failed due to timeout: {e}. "
                    f"traj_id: {payload.get('trajectory_ids', [])}. Retrying..."
                )
                resp_data = {"observations": [f"Tool call timeout after {timeout_seconds} seconds."], "dones": [1], "valids": [0], "processing_time_ms": timeout_seconds * 1000.0, "success": False}

            except aiohttp.ClientConnectorError as e:
                logging.error(
                    f"Attempt {attempt + 1} failed to connect to tool server: {e}. "
                    f"traj_id: {payload.get('trajectory_ids', [])}. Retrying..."
                )
                resp_data = {"observations": [f"Tool server connection error: {e}."], "dones": [1], "valids": [0], "processing_time_ms": 0.0, "success": False}

            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1} failed with error: {e}. "
                    f"traj_id: {payload.get('trajectory_ids', [])}. Retrying..."
                )
                resp_data = {"observations": [f"Tool call error: {e}."], "dones": [1], "valids": [0], "processing_time_ms": 0.0, "success": False}
            # if attempt == max_retries:
            #     raise
            await asyncio.sleep(1)
        session_request_time = time.time() - start_time
        resp_data['session_request_time'] = session_request_time
        resp_data['time awaiting response_ms'] = resp_data.get('time awaiting response_ms', 0.0)
        return resp_data
        
                
    async def send_batch_requests_async(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Robust version with retry logic"""
        safe_payload = sanitize_request(batch_data)
        
        # Debug: Check payload size
        try:
            json_str = json.dumps(safe_payload)
            payload_size_mb = len(json_str) / (1024 * 1024)
            if payload_size_mb > 1.0:
                logger.warning(f"Large payload detected: {payload_size_mb:.2f} MB")
        except Exception:
            pass

        try:
            return await self._aiohttp_request(safe_payload)
        except Exception as e:
            # Log error with context
            logging.error(f"Failed to send batch request after all retries: {e}")
            logging.error(f"Payload size: {len(str(safe_payload))} chars")
            
            # Save error data for debugging
            if not os.path.exists('tmp'):
                os.mkdir('tmp')  # Ensure tmp directory exists
            error_file = f"tmp/error_data_{uuid.uuid4().hex[:8]}.json"
            with open(error_file, 'w') as f:
                json.dump(safe_payload, f, indent=2)
            logging.error(f"Error data saved to {error_file} for debugging.")
            
            raise ValueError(f"Tool server communication failed: {e}")
    
    async def interact_with_tool_server(
        self,
        traj_id: str,
        action: str,
        do_action: bool,
        extra_fields: Any = None,
        is_last_step: bool = False,
    ) -> list[str]:
        batch_data = {
            "trajectory_ids": [traj_id],
            "actions": [action],
            "finish": [not do_action],
            "is_last_step": [is_last_step],
        }
        if extra_fields is not None:
            batch_data['extra_fields'] = [extra_fields]
        
        response = await self.send_batch_requests_async(batch_data)
        obs = response['observations'][0]
        done = int(response['dones'][0])
        is_valid_action = int(response['valids'][0])
        processing_time_ms = float(response.get('processing_time_ms', 0.0))
        success = response.get('success', True)
        
        
        # postprocess next_obs. For now we support two types of observations:
        # 1. string observations, which will be the most common case
        # 2. dict observations, e.g. {"obs": "some observation", "reward": 1.0}
        #     for now we only support "obs" and "reward" keys, but can be extended later
        tool_interact_info = {}
        if isinstance(obs, str):
            # can be invalid
            obs_text = obs
            reward = None
            tool_interact_info['obs'] = obs
            tool_interact_info['reward'] = None
        elif isinstance(obs, dict):
            assert "obs" in obs, f"Observation dict must contain 'obs' key, but got {obs.keys()}"
            obs_text = obs.get('obs', '')
            reward = obs.get('reward', None)
            assert isinstance(obs_text, str), f"Expected 'obs' to be a string, but got {type(obs)}"
            assert reward is None or isinstance(reward, (int, float)), f"Expected 'reward' to be None, int, or float, but got {type(reward)}"
            # store tool interaction info if exists
            tool_interact_info = {k: v for k, v in obs.items()} # image or other info
        else:
            raise ValueError(f"Invalid observation type: {type(obs)}. Expected str or dict.")
        
        tool_interact_info['obs'] = obs_text[:MAX_OBS_LENGTH]
        tool_interact_info['reward'] = reward
        tool_interact_info['trajectory_id'] = traj_id
        tool_interact_info['action'] = action
        tool_interact_info['is_last_step'] = is_last_step
        tool_interact_info['done'] = done
        tool_interact_info['valid_action'] = is_valid_action
        tool_interact_info['finish'] = not do_action
        tool_interact_info['invalid_reason'] = tool_interact_info.get('invalid_reason', None)
        tool_interact_info['processing_time_ms'] = processing_time_ms
        tool_interact_info['success'] = success
        
        for key in response.keys():
            if key not in ['observations', 'dones', 'valids', 'processing_time_ms', 'success'] and isinstance(response[key], float):
                tool_interact_info[key] = response[key]
        return tool_interact_info
        

    async def close_traj_tool_threads(
        self,
        request_id: str,
    ):
        """
            This function is used to close the trajectories that are overlong and clean up the tool server for corresponding tool threads.
        """
        finishs = [True] # all trajectories are finished
        actions = [''] # no actions, just finish the trajectories
        is_last_step = True # this is the last step
        batch_data = {
            "trajectory_ids": [request_id],
            "actions": actions,
            "finish": finishs, # if do_action is False, then it is a finish action, finishing the trajectory,
            "is_last_step": [is_last_step] * len(finishs)
        }
        _ = await self.send_batch_requests_async(batch_data)
        return 
        
        
    # async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    #     request_id = str(uuid4().hex)
        
    #     log_dir = str(project_root/f"RL/logs/verltool_agent_loop/{time_str}")
    #     os.makedirs(log_dir, exist_ok=True) 
    #     log_filename = os.path.join(log_dir, f"agent_traj_{request_id[-6:]}.txt")
        
    #     prompt_ids = list(kwargs["raw_prompt_ids"])
    #     multi_modal_data = kwargs.get("multi_modal_data") or {}
    #     image_data = multi_modal_data.get("image")
    #     encoded_image_data = [encode_image_url(img) for img in image_data] if image_data is not None else None
    #     audio_data = multi_modal_data.get("audio")
    #     encoded_audio_data = [encode_audio_data(audio) for audio in audio_data] if audio_data is not None else None
    #     use_tool = kwargs.get("use_tool", self.agent_config.enable_agent)
        
    #     metrics = {}
    #     # request_id = str(uuid4().hex)

    #     # # ================= [新增日志：确认最原始的训练输入数据] =================        
    #     # # 直接解码刚刚拿到的 raw_prompt_ids
    #     # absolute_raw_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        
    #     # with open(log_filename, "a", encoding="utf-8") as f:
    #     #     f.write(f"\n{'='*20} [INIT: {request_id[-6:]}] 绝对原始 Prompt (从 kwargs 传入) {'='*20}\n")
    #     #     f.write(absolute_raw_text + "\n")
    #     #     f.write(f"{'='*80}\n\n")
    #     # # =====================================================================
        
    #     stats_dict = {
    #         "num_turns": 0,
    #         "empty_responses": 0,
    #         "valid_action": 0,
    #         "action_lengths": [],
    #         "action_logps": [],
    #         "obs_lengths": [],
    #         "rewards": [],
    #         "tool_interact_info": [],
    #         "is_traj_finished": False,
    #         "valid_traj": 1,
    #         "retokenization_diff": []
    #     }
        
    #     if image_data or audio_data:
    #         raw_prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
    #         processor_kwargs = {"text": [raw_prompt_text], "return_tensors": "pt"}
    #         if image_data:
    #             processor_kwargs["images"] = image_data
    #         if audio_data:
    #             processor_kwargs["audio"] = audio_data
    #         model_inputs = self.processor(**processor_kwargs)
    #         prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        
    #     max_turns = self.max_turns if not kwargs.get("validate", False) else self.val_max_turns
    #     max_response_length = self.train_max_response_length if not kwargs.get("validate", False) else self.val_max_response_length
    #     max_action_length = self.max_action_length or max_response_length
    #     max_obs_length = self.max_obs_length
        
    #     agent_sampling_params = sampling_params.copy()
    #     agent_sampling_params.update({
    #         "n": 1,  # already repeated by n times in repeat_inputs_by_n
    #         "stop": self.action_stop_tokens,  # stop when generated an end of action
    #         "include_stop_str_in_output": True,
    #         "logprobs": self.logprobs,
    #     })
        
    #     running_prompt_ids = prompt_ids.copy()
    #     running_image_data = image_data.copy() if image_data is not None else None
    #     running_audio_data = audio_data.copy() if audio_data is not None else None
    #     response_mask = []
    #     response_logprobs = []
    #     traj_stop_reason = ""

    #     logger.debug(f"Starting agent loop for traj_id={request_id} with use_tool={use_tool}, max_turns={max_turns}, max_response_length={max_response_length}, max_action_length={max_action_length}, max_obs_length={max_obs_length}")
        
    #     # ================= [新增日志：记录初始原始 Prompt] =================       
    #     # 解码初始 Prompt 文本
    #     initial_prompt_text = self.tokenizer.decode(running_prompt_ids, skip_special_tokens=False)
        
    #     with open(log_filename, "a", encoding="utf-8") as f:
    #         f.write(f"\n{'#'*30} [TRAJECTORY START: {request_id}] {'#'*30}\n")
    #         f.write(f"🚀 [Initial Prompt]:\n")
    #         f.write(initial_prompt_text + "\n")
    #         f.write(f"{'#'*80}\n\n")
    #     # =================================================================

    #     for step in range(max_turns + 1):
    #         previous_length = len(running_prompt_ids)
    #         available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
    #         max_tokens_for_this_turn = min(max_action_length, available_length)
    #         is_last_step = (step == max_turns)
    #         if max_tokens_for_this_turn <= 0:
    #             traj_stop_reason = "max_model_len_exceeded"
    #             break
    #         agent_sampling_params["max_tokens"] = max_tokens_for_this_turn # for vllm
    #         logger.debug(f"Turn {step}: available_length={available_length}, max_tokens_for_this_turn={max_tokens_for_this_turn}")
    #         # agent_sampling_params["max_new_tokens"] = max_tokens_for_this_turn # for sglang
    #         with simple_timer("generate_sequences", metrics):
    #             output = await self.server_manager.generate(
    #                 request_id=request_id,
    #                 prompt_ids=running_prompt_ids,
    #                 sampling_params=agent_sampling_params,
    #                 image_data=running_image_data,
    #                 audio_data=running_audio_data,
    #             ) # request_id here should be unique for each generate call, otherwise vllm can generate empty response
    #             if output.text.strip() == "":
    #                 logger.warning(f"Turn {step}: Generated empty response for traj_id={request_id}. prompt_ids length: {len(running_prompt_ids)}")
    #         gen_ids = output.token_ids
    #         gen_logprobs = output.log_probs or [0.0] * len(gen_ids)
    #         gen_text = output.text

    #         # ================= [新增日志：写入文件记录模型输出] =================            
    #         with open(log_filename, "a", encoding="utf-8") as f:
    #             f.write(f"\n{'='*20} [Step {step}] 🤖 模型生成 {'='*20}\n")
    #             f.write(gen_text + "\n")
    #         # =================================================================

    #         running_prompt_ids.extend(gen_ids)
    #         response_mask.extend([1] * len(gen_ids))
    #         response_logprobs.extend(gen_logprobs)
            
    #         stats_dict["num_turns"] += 1
    #         stats_dict["action_lengths"].append(len(gen_ids))
    #         stats_dict["action_logps"].append(np.mean(gen_logprobs) if len(gen_logprobs) > 0 else 0.0)
    #         if gen_text.strip() == "":
    #             stats_dict["empty_responses"] += 1
            
    #         finish_reason = output.finish_reason
    #         stop_reason = output.stop_reason
    #         if isinstance(stop_reason, int):
    #             print(f"Turn {step}: stop_reason is int: {stop_reason}")
    #             stop_reason = self.tokenizer.decode([stop_reason])[0]
            
    #         if not use_tool:
    #             # only one turn generation
    #             stats_dict["is_traj_finished"] = True
    #             traj_stop_reason = f"no_tool-{finish_reason}"
    #             break
            
    #         # judge whether to interact with tool or finish
    #         do_action = False
    #         action_text = ""

    #         # 不再强依赖 finish_reason == "stop"，只要输出里包含了提取符就算作动作
    #         for ext_token in self.action_extract_tokens:
    #             if ext_token in gen_text:
    #                 do_action = True
                    
    #                 # 核心优化：使用 rfind 找到最后一次出现 ext_token 的位置
    #                 # 比如模型输出: "思考过程... ```click[id=1]```"
    #                 # ext_token 是 "```"
    #                 last_idx = gen_text.rfind(ext_token)
                    
    #                 # 截取到最后那个 ``` 的末尾
    #                 action_text = gen_text[:last_idx + len(ext_token)]
    #                 break

    #         # ================= [新增日志：写入模型输出动作指令] =================               
    #         with open(log_filename, "a", encoding="utf-8") as f:
    #             f.write(f"\n{'='*20} [Step {step}] 🔧 动作截取 {'='*20}\n")
    #             f.write(action_text if action_text.strip() else "[警告：生成了空动作！]\n")
    #         # =================================================================

    #         # send generated action to tool server
    #         if do_action and not is_last_step:
    #             extra_fields = kwargs.get("extra_info", {}).copy()
    #             # if encoded_image_data is not None:
    #             #     extra_fields["images"] = encoded_image_data
    #             # if encoded_audio_data is not None:
    #             #     extra_fields["audio"] = encoded_audio_data
    #             logger.info(f"Turn {step}: finish_reason={finish_reason}, stop_reason={stop_reason}, do_action={do_action}, action_text={json.dumps(action_text[-50:])}")
    #             start = metrics.get("tool_calls", 0.0)
    #             with simple_timer("tool_calls", metrics):
    #                 tool_results = await self.interact_with_tool_server(
    #                     traj_id=request_id,
    #                     action=action_text,
    #                     do_action=do_action,
    #                     extra_fields=extra_fields,
    #                     is_last_step=is_last_step,
    #                 )
    #             end = metrics.get("tool_calls", 0.0)
    #             tool_results['interact_time_ms'] = (end - start) * 1000.0
                
    #             # process observations and prepare for next turn
    #             obs_text = tool_results['obs']

    #             # ================= [新增日志：写入文件记录工具反馈] =================               
    #             with open(log_filename, "a", encoding="utf-8") as f:
    #                 f.write(f"\n{'='*20} [Step {step}] 🔧 工具返回 {'='*20}\n")
    #                 f.write(obs_text if obs_text.strip() else "[警告：工具返回了空内容！]\n")
    #             # =================================================================


    #             # =====================================================================
    #             # [新增模块] 视觉上下文压缩拦截器 (Observation-Level Compression)
    #             # 作用：将过长的纯文本观测渲染为图片，极大地节省 Context Window Token
    #             # =====================================================================
                
    #             COMPRESSION_THRESHOLD = 0 
                
    #             if getattr(self.agent_config, 'enable_obs_compression', True) and len(obs_text) > COMPRESSION_THRESHOLD:
    #                 saved_dir = str(project_root/f"RL/logs/verltool_agent_loop/{time_str}/obs_img")
    #                 os.makedirs(saved_dir, exist_ok=True)
                    
    #                 # 1. 异步执行 CPU 密集型渲染，防止卡死事件循环
    #                 loop = asyncio.get_event_loop()
    #                 compressed_img, _ = await loop.run_in_executor(
    #                     None, 
    #                     self.vtc.render_text_to_image, 
    #                     obs_text
    #                 )
                    
    #                 # 异步保存图片 (加入 step 防止覆盖)
    #                 img_path = f"{saved_dir}/compressed_obs_{request_id[-6:]}_step{step}.png"
    #                 await loop.run_in_executor(None, compressed_img.save, img_path)
                    
    #                 # 2. 将图片转换为原生框架支持的 Base64 格式，防止 decode_image_url 崩溃
    #                 import io
    #                 import base64
    #                 buffered = io.BytesIO()
    #                 compressed_img.save(buffered, format="PNG")
    #                 img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #                 img_data_uri = f"data:image/png;base64,{img_b64_str}"
                    
    #                 # 3. 安全注入到 tool_results
    #                 if 'image' not in tool_results or tool_results['image'] is None:
    #                     tool_results['image'] = []
    #                 elif not isinstance(tool_results['image'], list):
    #                     tool_results['image'] = [tool_results['image']]
                        
    #                 tool_results['image'].append(img_data_uri) # 传入 URI 而不是 PIL 对象
                    
    #                 # 4. 替换文本观测
    #                 obs_text = f"System Observation (Compressed): <image>"
    #                 tool_results['obs'] = obs_text 
                    
    #                 logger.info(f"Turn {step}: Compressed long observation into an image. Saved to {img_path}")
    #             # =====================================================================

    #             if tool_results.get('image', None):
    #                 images = [tool_results['image']] if not isinstance(tool_results['image'], list) else tool_results['image']
    #                 decoded_images = [decode_image_url(img_url) for img_url in images]

    #                 if running_image_data is None:
    #                     running_image_data = []
                        
    #                 running_image_data.extend(decoded_images)
    #                 # add image placeholder token ids
    #                 # first see whether there are <image> tags in obs_text
    #                 num_image_tags = obs_text.count("<image>")
    #                 if num_image_tags < len(decoded_images):
    #                     # append at the end
    #                     obs_text += "<image>" * (len(decoded_images) - num_image_tags)
    #                     num_image_tags = len(decoded_images)
    #                 # now replace <image> tags with image placeholder tokens
    #                 obs_text = obs_text.replace("<image>", self.qwen_image_placeholder, num_image_tags)
    #                 # obs_token_ids = self.processor(text=[obs_text], images=decoded_images, return_tensors="pt")["input_ids"].squeeze(0).tolist()

    #                 # --- 新增：安全检查 processor 是否存在 ---
    #                 if self.processor is not None:
    #                     # 多模态模型：使用 processor 处理图文
    #                     obs_token_ids = self.processor(text=[obs_text], images=decoded_images, return_tensors="pt")["input_ids"].squeeze(0).tolist()
    #                 else:
    #                     # 纯文本模型：忽略图片，直接对文本进行 tokenize
    #                     obs_token_ids = self.tokenizer.encode(obs_text)
    #                 # --------------------------------------

    #                 if max_obs_length < len(obs_token_ids):
    #                     if self.agent_config.truncate_obs_side == 'left':
    #                         truncation_index = max_obs_length
    #                         if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
    #                             # find the nearest non-truncate token id before max_obs_length
    #                             truncation_index = max_obs_length - 1
    #                             while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
    #                                 truncation_index -= 1
    #                         obs_token_ids = obs_token_ids[:truncation_index]
    #                         obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
    #                     else:
    #                         raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
    #             elif tool_results.get('audio', None):
    #                 audios = [tool_results['audio']] if not isinstance(tool_results['audio'], list) else tool_results['audio']
    #                 decoded_audios = [decode_image_url(audio_url) for audio_url in audios]
    #                 running_audio_data.extend(decoded_audios)
    #                 # add audio placeholder token ids
    #                 # first see whether there are <audio> tags in obs_text
    #                 num_audio_tags = obs_text.count("<audio>")
    #                 if num_audio_tags < len(decoded_audios):
    #                     # append at the end
    #                     obs_text += "<audio>" * (len(decoded_audios) - num_audio_tags)
    #                     num_audio_tags = len(decoded_audios)
    #                 # now replace <audio> tags with audio placeholder tokens
    #                 obs_text = obs_text.replace("<audio>", self.qwen_audio_placeholder, num_audio_tags)
    #                 # for mtrl
    #                 if self.enable_mtrl:
    #                     obs_text = self.mtrl_sep.format(obs=obs_text)
    #                 obs_token_ids = self.processor(text=[obs_text], audio=decoded_audios, return_tensors="pt")["input_ids"].squeeze(0).tolist()
    #                 # obs_token_ids = obs_token_ids[:max_obs_length]
    #                 if max_obs_length < len(obs_token_ids):
    #                     if self.agent_config.truncate_obs_side == 'left':
    #                         truncation_index = max_obs_length
    #                         if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
    #                             # find the nearest non-truncate token id before max_obs_length
    #                             truncation_index = max_obs_length - 1
    #                             while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
    #                                 truncation_index -= 1
    #                         obs_token_ids = obs_token_ids[:truncation_index]
    #                         obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
    #                     else:
    #                         raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
    #             else:
    #                 obs_token_ids = self.tokenizer.encode(obs_text)
    #                 if max_obs_length < len(obs_token_ids):
    #                     if self.agent_config.truncate_obs_side == 'left':
    #                         obs_token_ids = obs_token_ids[-max_obs_length:]
    #                         obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
    #                     elif self.agent_config.truncate_obs_side == 'right':
    #                         obs_token_ids = obs_token_ids[:max_obs_length]
    #                         obs_token_ids.extend(self.tokenizer.encode("(truncated)..."))
    #                     elif self.agent_config.truncate_obs_side == 'middle':
    #                         half_len = max_obs_length // 2
    #                         obs_token_ids = obs_token_ids[:half_len] + self.tokenizer.encode("...(truncated)...") + obs_token_ids[-half_len:]
    #                     else:
    #                         raise ValueError(f"Invalid truncate_obs_side: {self.agent_config.truncate_obs_side}")
                
    #             if self.enable_mtrl:
    #                 obs_token_ids = self.mtrl_sep_prefix_ids + obs_token_ids + self.mtrl_sep_suffix_ids
    #             # update stats
    #             stats_dict["obs_lengths"].append(len(obs_token_ids))
    #             if 'reward' in tool_results and tool_results['reward'] is not None:
    #                 stats_dict["rewards"].append(tool_results['reward'] if 'reward' in tool_results else 0.0)
    #             stats_dict["valid_action"] += tool_results['valid_action'] if 'valid_action' in tool_results else 0
    #             stats_dict["tool_interact_info"].append(tool_results)
                
    #             # update running prompt
    #             running_prompt_ids.extend(obs_token_ids)
    #             if self.agent_config.mask_observations:
    #                 response_mask.extend([0] * len(obs_token_ids))
    #             else:
    #                 response_mask.extend([1] * len(obs_token_ids))
    #             response_logprobs.extend([0.0] * len(obs_token_ids)) # pad with 0.0 logprobs for observations
                
    #             if self.agent_config.retokenization:
    #                 new_text = self.tokenizer.decode(running_prompt_ids[previous_length:], skip_special_tokens=False)
    #                 new_ids = self.tokenizer.encode(new_text)
    #                 if new_ids != running_prompt_ids[previous_length:]:
    #                     stats_dict['retokenization_diff'].append(1)
    #                     logger.debug(f"Retokenization changed the length from {len(running_prompt_ids) - previous_length} to {len(new_ids)}. traj_id={request_id}, turn={step}")
    #                 else:
    #                     stats_dict['retokenization_diff'].append(0)
    #                 running_prompt_ids = running_prompt_ids[:previous_length] + new_ids
    #                 if len(response_mask) > (len(running_prompt_ids) - len(prompt_ids)):
    #                     response_mask = response_mask[:len(running_prompt_ids) - len(prompt_ids)]
    #                 else:
    #                     response_mask.extend([response_mask[-1]] * (len(running_prompt_ids) - len(prompt_ids) - len(response_mask)))
    #                 if len(response_logprobs) > (len(running_prompt_ids) - len(prompt_ids)):
    #                     response_logprobs = response_logprobs[:len(running_prompt_ids) - len(prompt_ids)]
    #                 else:
    #                     response_logprobs.extend([0.0] * (len(running_prompt_ids) - len(prompt_ids) - len(response_logprobs)))

    #             if tool_results['done']:
    #                 # finish the trajectory
    #                 stats_dict["is_traj_finished"] = True
    #                 traj_stop_reason = "tool_signaled_done"
    #                 break

    #         else:
    #             # finish the trajectory
    #             if finish_reason == "stop":
    #                 if not stop_reason:
    #                     stats_dict["is_traj_finished"] = True
    #                     if gen_text.strip() == "":
    #                         traj_stop_reason = "model_chose_to_finish_with_empty_response"
    #                     else:
    #                         traj_stop_reason = "model_chose_to_finish"
    #                 else:
    #                     stats_dict["is_traj_finished"] = False
    #                     if do_action and is_last_step:
    #                         traj_stop_reason = "max_turns_reached"
    #                     else:
    #                         traj_stop_reason = "no_action_stop_token_found_in_stop_reason=" + stop_reason
    #             elif finish_reason == "length":
    #                 stats_dict["is_traj_finished"] = False
    #                 traj_stop_reason = "max_response_length_reached"
    #             else:
    #                 stats_dict["is_traj_finished"] = True
    #                 traj_stop_reason = f"finish_reason_{finish_reason}"
    #             break
        
    #     logger.debug(f"Trajectory {request_id} finished after {step} turns. Stop reason: {traj_stop_reason}")
    #     response_ids = running_prompt_ids[len(prompt_ids):]
    #     assert len(response_ids) == len(response_mask), f"Response ids and mask length mismatch: {len(response_ids)} vs {len(response_mask)}"
    #     start = time.time()
    #     # let close traj run in background but don't wait
    #     asyncio.create_task(self.close_traj_tool_threads(request_id=request_id))
    #     # await self.close_traj_tool_threads(request_id=request_id)
    #     end = time.time()
    #     close_traj_time = end - start
        
    #     if self.agent_config.mask_overlong_loss and not stats_dict["is_traj_finished"]:
    #         # mask the whole response
    #         response_mask = [0] * len(response_mask)
    #         stats_dict["valid_traj"] = 0
    #         logger.info(f"Masking the whole response for traj_id={request_id} due to overlong trajectory and not finished.")
        
        
    #     if self.agent_config.mask_void_traj:
    #         response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    #         has_answer = "\\boxed" in response_text or "final_answer(" in response_text
    #         if stats_dict["valid_action"] == 0 and not has_answer:
    #             # mask the whole response
    #             response_mask = [0] * len(response_mask)
    #             stats_dict["valid_traj"] = 0
    #             logger.info(f"Masking the whole response for traj_id={request_id} due to void trajectory (no valid action or no final answer). valid_action={stats_dict['valid_action']}, has_answer={has_answer}")
            
    #     verl_tool_metrics = {
    #         "num_turns": stats_dict["num_turns"],
    #         "empty_responses": stats_dict["empty_responses"],
    #         "valid_action": stats_dict["valid_action"],
    #         "per_action_length": np.mean(stats_dict["action_lengths"]) if len(stats_dict["action_lengths"]) > 0 else 0,
    #         "per_obs_length": np.mean(stats_dict["obs_lengths"]) if len(stats_dict["obs_lengths"]) > 0 else 0,
    #         "per_action_logp": np.mean(stats_dict["action_logps"]) if len(stats_dict["action_logps"]) > 0 else 0,
    #         "per_reward_from_tool": np.mean(stats_dict["rewards"]) if len(stats_dict["rewards"]) > 0 else 0,
    #         "tool_call_success": np.mean([info.get("success", 1.0) for info in stats_dict["tool_interact_info"]]) if len(stats_dict["tool_interact_info"]) > 0 else 1.0,
    #         "traj_actions_length": sum(stats_dict["action_lengths"]),
    #         "traj_obs_length": sum(stats_dict["obs_lengths"]),
    #         "generated_length": len(output.token_ids),
    #         "is_traj_finished": float(stats_dict["is_traj_finished"]),
    #         "valid_traj": float(stats_dict["valid_traj"]),
    #         "no_loss_on_traj": response_mask.count(1) == 0,
    #         "retokenization_diff": np.mean(stats_dict["retokenization_diff"]) if len(stats_dict["retokenization_diff"]) > 0 else 0,
    #         "tool_processing_time_sec": sum(info.get("processing_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "tool_queue_time_sec": sum(info.get("queue_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "client_queued_time_sec": sum(info.get("client_queued_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "client_conn_create_time_sec": sum(info.get("client_conn_create_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "client_request_send_time_sec": sum(info.get("client_request_send_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "response_read_time_sec": sum(info.get("response_read_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "response_size_mb": sum(info.get("response_size_mb", 0.0) for info in stats_dict["tool_interact_info"]),
    #         "response_await_time_sec": sum(info.get("time awaiting response_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "interact_time_time_sec": sum(info.get("interact_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
    #         "session_request_time": sum(info.get("session_request_time", 0.0) for info in stats_dict["tool_interact_info"]),
    #         "X-Process-Time_sec": sum(info.get("X-Process-Time", 0.0) for info in stats_dict["tool_interact_info"]),
    #         "close_traj_time": close_traj_time,
    #     }
    #     # additional metrics in tool_interact_info can be added later
    #     if stats_dict["tool_interact_info"] and all("metrics" in info for info in stats_dict["tool_interact_info"]):
    #         # do mean aggregation for each metric key
    #         tool_metric_keys = stats_dict["tool_interact_info"][0]["metrics"].keys()
    #         for key in tool_metric_keys:
    #             try:
    #                 verl_tool_metrics[f"tool_avg_{key}"] = np.mean([float(info["metrics"][key]) for info in stats_dict["tool_interact_info"] if key in info["metrics"]])
    #             except Exception as e:
    #                 logger.warning(f"Failed to compute mean for tool metric {key}: {e}")
    #     # additional per-turn logp
    #     for i, logp in enumerate(stats_dict["action_logps"]):
    #         verl_tool_metrics[f"turn_{i+1}_action_logp"] = logp
        
    #     # Safely truncate final response so that we do not cut through multimodal placeholder tokens.
    #     # Here we treat `cut_index` as "the first index to be dropped", so the last kept index is `cut_index - 1`.
    #     if len(response_ids) > self.response_length:
    #         cut_index = self.response_length
    #         last_keep = cut_index - 1
    #         if last_keep >= 0 and response_ids[last_keep] in self.non_truncate_token_ids:
    #             # Step backwards over a contiguous block of placeholder tokens so that
    #             # we either keep the whole block (if it is fully within the limit) or
    #             # drop it entirely when it would otherwise be partially kept.
    #             while last_keep >= 0 and response_ids[last_keep] in self.non_truncate_token_ids:
    #                 last_keep -= 1
    #             cut_index = last_keep + 1
    #         response_ids = response_ids[:cut_index]
    #         response_mask = response_mask[:cut_index]
    #         response_logprobs = response_logprobs[:cut_index]

    #     # Keep image features aligned with tokens to avoid tokens/features mismatch
    #     # errors in models such as Qwen3-VL.
    #     # Count only the tokens that will actually be fed to the model
    #     # (prompt_ids + truncated response_ids), and approximate available image
    #     # segments by counting contiguous token: <|vision_start|>
    #     if running_image_data is not None:
    #         full_ids = prompt_ids + response_ids
    #         vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    #         num_visual_segments = full_ids.count(vision_start_id)
    #         if len(running_image_data) > num_visual_segments:
    #             running_image_data = running_image_data[:num_visual_segments]
        
    #     multi_modal_output = {}
    #     if running_image_data is not None:
    #         multi_modal_output["image"] = running_image_data
    #     if running_audio_data is not None:
    #         multi_modal_output["audio"] = running_audio_data

    #     output = AgentLoopOutput(
    #         prompt_ids=prompt_ids,
    #         response_ids=response_ids[: self.response_length],
    #         response_mask=response_mask[: self.response_length],
    #         response_logprobs=response_logprobs[: self.response_length],
    #         multi_modal_data=multi_modal_output,
    #         num_turns=stats_dict["num_turns"],
    #         metrics=metrics,
    #         extra_fields={"tool_interact_info": stats_dict.get("tool_interact_info", []), "traj_stop_reason": traj_stop_reason, "verl_tool_metrics": verl_tool_metrics},
    #     )
    #     return output


    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        request_id = str(uuid4().hex)
        
        log_dir = self._rollout_log_dir(self.config)
        os.makedirs(log_dir, exist_ok=True) 
        rollout_global_step = int(kwargs.get("_rollout_global_step", -1))
        rollout_epoch = self._rollout_log_epoch(
            self.config,
            rollout_global_step,
            kwargs.get("_rollout_epoch"),
            kwargs.get("validate", False),
        )
        log_filename = self._rollout_log_path(log_dir, rollout_epoch, kwargs.get("validate", False))
        log_context = {
            "trajectory_id": request_id,
            "trajectory_suffix": request_id[-6:],
            "run_started_at": time_str,
            "project_name": self.config.trainer.get("project_name", "unknown_project"),
            "experiment_name": self.config.trainer.get("experiment_name", "unknown_experiment"),
            "global_step": rollout_global_step,
            "epoch": rollout_epoch,
            "sample_index": kwargs.get("_rollout_sample_index"),
            "rollout_n": kwargs.get("_rollout_n"),
            "validate": kwargs.get("validate", False),
        }
        
        prompt_ids = list(kwargs["raw_prompt_ids"])
        multi_modal_data = kwargs.get("multi_modal_data") or {}
        image_data = multi_modal_data.get("image")
        encoded_image_data = [encode_image_url(img) for img in image_data] if image_data is not None else None
        audio_data = multi_modal_data.get("audio")
        encoded_audio_data = [encode_audio_data(audio) for audio in audio_data] if audio_data is not None else None
        use_tool = kwargs.get("use_tool", self.agent_config.enable_agent)
        
        metrics = {}

        # 预先初始化一个空的 output 对象，防止因为提前 break 导致后面引用报错
        class DummyOutput:
            def __init__(self):
                self.token_ids = []
        output = DummyOutput()
        
        stats_dict = {
            "num_turns": 0,
            "empty_responses": 0,
            "valid_action": 0,
            "action_lengths": [],
            "action_logps": [],
            "obs_lengths": [],
            "rewards": [],
            "tool_interact_info": [],
            "is_traj_finished": False,
            "valid_traj": 1,
            "retokenization_diff": []
        }

        # ================= [新增模块: 初始化 Token 统计字典] =================
        token_stats = {
            "initial_prompt_tokens": len(prompt_ids),
            "total_gen_tokens": 0,
            "total_original_obs_text_tokens": 0, # 压缩前的原始纯文本 obs 长度（理论开销）
            "total_actual_obs_tokens": 0,        # 实际喂给模型环境的总 token（实际开销）
            "total_obs_image_tokens": 0,         # 视觉占用的 token
            "total_obs_audio_tokens": 0          # 音频占用的 token
        }
        # =====================================================================
        
        if image_data or audio_data:
            raw_prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            processor_kwargs = {"text": [raw_prompt_text], "return_tensors": "pt"}
            if image_data:
                processor_kwargs["images"] = image_data
            if audio_data:
                processor_kwargs["audio"] = audio_data
            model_inputs = self.processor(**processor_kwargs)
            prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
            # 更新经过 processor 处理后的真实 prompt 长度（包含多模态 token）
            token_stats["initial_prompt_tokens"] = len(prompt_ids)
        
        max_turns = self.max_turns if not kwargs.get("validate", False) else self.val_max_turns
        max_response_length = self.train_max_response_length if not kwargs.get("validate", False) else self.val_max_response_length
        max_action_length = self.max_action_length or max_response_length
        max_obs_length = self.max_obs_length
        
        agent_sampling_params = sampling_params.copy()
        agent_sampling_params.update({
            "n": 1,
            "stop": self.action_stop_tokens,
            "include_stop_str_in_output": True,
            "logprobs": self.logprobs,
        })
        
        running_prompt_ids = prompt_ids.copy()
        running_image_data = image_data.copy() if image_data is not None else None
        running_audio_data = audio_data.copy() if audio_data is not None else None
        response_mask = []
        response_logprobs = []
        traj_stop_reason = ""

        logger.debug(f"Starting agent loop for traj_id={request_id} with use_tool={use_tool}, max_turns={max_turns}, max_response_length={max_response_length}, max_action_length={max_action_length}, max_obs_length={max_obs_length}")
        
        # ================= [新增日志：记录初始原始 Prompt] =================      
        initial_prompt_text = self.tokenizer.decode(running_prompt_ids, skip_special_tokens=False)
        self._append_rollout_jsonl(
            log_filename,
            log_context,
            "trajectory_start",
            {
                "initial_prompt": initial_prompt_text,
                "initial_prompt_tokens": token_stats["initial_prompt_tokens"],
                "use_tool": use_tool,
                "max_turns": max_turns,
                "max_response_length": max_response_length,
                "max_action_length": max_action_length,
                "max_obs_length": max_obs_length,
            },
        )
        # =================================================================
        
        # [修改点] 循环外的初始化：强行触发 Step 0 的环境交互 (Action 为空)
        action_text = ""
        do_action = True 

        for step in range(max_turns + 1):
            is_last_step = (step == max_turns)
            turn_start_length = len(running_prompt_ids)
            
            # =====================================================================
            # [获取环境观测 (Environment Interaction)]
            # =====================================================================
            if use_tool and do_action:
                extra_fields = kwargs.get("extra_info", {}).copy()
                logger.info(f"Turn {step}: interact with tool server. action_text={json.dumps(action_text[-50:])}")
                start = metrics.get("tool_calls", 0.0)
                
                with simple_timer("tool_calls", metrics):
                    tool_results = await self.interact_with_tool_server(
                        traj_id=request_id,
                        action=action_text,
                        do_action=do_action,
                        extra_fields=extra_fields,
                        is_last_step=False, 
                    )
                end = metrics.get("tool_calls", 0.0)
                tool_results['interact_time_ms'] = (end - start) * 1000.0
                
                obs_text = tool_results['obs']

                # ================= [新增模块: 截获压缩前的原始文本 Token] =================
                # 记录未经任何压缩的原始环境反馈长度，用于后续评估压缩率
                raw_text_len = len(self.tokenizer.encode(obs_text))
                token_stats["total_original_obs_text_tokens"] += raw_text_len
                # =====================================================================

                # ================= [新增日志：写入文件记录工具反馈] =================                
                self._append_rollout_jsonl(
                    log_filename,
                    log_context,
                    "environment_observation",
                    {
                        "turn": step,
                        "observation_type": "initial" if step == 0 else "tool",
                        "observation": obs_text if obs_text.strip() else "",
                        "empty_observation": not bool(obs_text.strip()),
                        "raw_observation_tokens": raw_text_len,
                        "tool_status": {
                            "done": tool_results.get("done"),
                            "valid_action": tool_results.get("valid_action"),
                            "reward": tool_results.get("reward"),
                            "success": tool_results.get("success"),
                            "processing_time_ms": tool_results.get("processing_time_ms"),
                            "interact_time_ms": tool_results.get("interact_time_ms"),
                            "response_size_mb": tool_results.get("response_size_mb"),
                        },
                    },
                )
                # =================================================================

                # =====================================================================
                # [视觉上下文压缩拦截器 (Observation-Level Compression)]
                # =====================================================================
                COMPRESSION_THRESHOLD = 0 
                if getattr(self.agent_config, 'enable_obs_compression', True) and len(obs_text) > COMPRESSION_THRESHOLD:
                    saved_dir = os.path.join(log_dir, "obs_img")
                    os.makedirs(saved_dir, exist_ok=True)
                    
                    loop = asyncio.get_event_loop()
                    
                    render_kwargs = {
                        "obs_text": obs_text,
                        "max_width": 1024,
                        "max_height": 2048,
                        "title": "--- SYSTEM OBSERVATION ---",
                        "use_compact_mode": True
                    }
                    raw_img, _ = await loop.run_in_executor(
                        None, 
                        lambda: self.vtc.render_text_to_image(**render_kwargs)
                    )
                    
                    COMPRESSION_FACTOR = 1.5 
                    compressed_img_list = await loop.run_in_executor(
                        None, 
                        self.vtc.compress_image_arrays, 
                        [raw_img], 
                        COMPRESSION_FACTOR
                    )
                    compressed_img = compressed_img_list[0]
                    
                    img_path = f"{saved_dir}/compressed_obs_{request_id[-6:]}_step{step}.png"
                    await loop.run_in_executor(None, compressed_img.save, img_path)
                    # self._append_rollout_jsonl(
                    #     log_filename,
                    #     log_context,
                    #     "observation_compressed",
                    #     {
                    #         "turn": step,
                    #         "image_path": img_path,
                    #         "compression_factor": COMPRESSION_FACTOR,
                    #         "raw_observation_tokens": raw_text_len,
                    #     },
                    # )
                    
                    import io
                    import base64
                    buffered = io.BytesIO()
                    compressed_img.save(buffered, format="PNG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    img_data_uri = f"data:image/png;base64,{img_b64_str}"
                    
                    if 'image' not in tool_results or tool_results['image'] is None:
                        tool_results['image'] = []
                    elif not isinstance(tool_results['image'], list):
                        tool_results['image'] = [tool_results['image']]
                        
                    tool_results['image'].append(img_data_uri) 
                    
                    obs_text = f"System Observation (Compressed): <image>"
                    tool_results['obs'] = obs_text 
                    logger.info(f"Turn {step}: Rendered high-res and compressed by {COMPRESSION_FACTOR}x. Saved to {img_path}")

                if tool_results.get('image', None):
                    images = [tool_results['image']] if not isinstance(tool_results['image'], list) else tool_results['image']
                    decoded_images = [decode_image_url(img_url) for img_url in images]

                    if running_image_data is None:
                        running_image_data = []
                        
                    running_image_data.extend(decoded_images)
                    num_image_tags = obs_text.count("<image>")
                    if num_image_tags < len(decoded_images):
                        obs_text += "<image>" * (len(decoded_images) - num_image_tags)
                        num_image_tags = len(decoded_images)
                        
                    obs_text = obs_text.replace("<image>", self.qwen_image_placeholder, num_image_tags)

                    if self.processor is not None:
                        obs_token_ids = self.processor(text=[obs_text], images=decoded_images, return_tensors="pt")["input_ids"].squeeze(0).tolist()
                    else:
                        obs_token_ids = self.tokenizer.encode(obs_text)

                    if max_obs_length < len(obs_token_ids):
                        if self.agent_config.truncate_obs_side == 'left':
                            truncation_index = max_obs_length
                            if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
                                truncation_index = max_obs_length - 1
                                while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
                                    truncation_index -= 1
                            obs_token_ids = obs_token_ids[:truncation_index]
                            obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                        else:
                            raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
                            
                elif tool_results.get('audio', None):
                    audios = [tool_results['audio']] if not isinstance(tool_results['audio'], list) else tool_results['audio']
                    decoded_audios = [decode_image_url(audio_url) for audio_url in audios]
                    running_audio_data.extend(decoded_audios)
                    num_audio_tags = obs_text.count("<audio>")
                    if num_audio_tags < len(decoded_audios):
                        obs_text += "<audio>" * (len(decoded_audios) - num_audio_tags)
                        num_audio_tags = len(decoded_audios)
                        
                    obs_text = obs_text.replace("<audio>", self.qwen_audio_placeholder, num_audio_tags)
                    if self.enable_mtrl:
                        obs_text = self.mtrl_sep.format(obs=obs_text)
                    obs_token_ids = self.processor(text=[obs_text], audio=decoded_audios, return_tensors="pt")["input_ids"].squeeze(0).tolist()
                    
                    if max_obs_length < len(obs_token_ids):
                        if self.agent_config.truncate_obs_side == 'left':
                            truncation_index = max_obs_length
                            if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
                                truncation_index = max_obs_length - 1
                                while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
                                    truncation_index -= 1
                            obs_token_ids = obs_token_ids[:truncation_index]
                            obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                        else:
                            raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
                else:
                    obs_token_ids = self.tokenizer.encode(obs_text)
                    if max_obs_length < len(obs_token_ids):
                        if self.agent_config.truncate_obs_side == 'left':
                            obs_token_ids = obs_token_ids[-max_obs_length:]
                            obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                        elif self.agent_config.truncate_obs_side == 'right':
                            obs_token_ids = obs_token_ids[:max_obs_length]
                            obs_token_ids.extend(self.tokenizer.encode("(truncated)..."))
                        elif self.agent_config.truncate_obs_side == 'middle':
                            half_len = max_obs_length // 2
                            obs_token_ids = obs_token_ids[:half_len] + self.tokenizer.encode("...(truncated)...") + obs_token_ids[-half_len:]
                        else:
                            raise ValueError(f"Invalid truncate_obs_side: {self.agent_config.truncate_obs_side}")
                
                if self.enable_mtrl:
                    obs_token_ids = self.mtrl_sep_prefix_ids + obs_token_ids + self.mtrl_sep_suffix_ids
                    
                stats_dict["obs_lengths"].append(len(obs_token_ids))

                # ================= [新增模块: 统计环境观测 Token 花费] =================
                actual_obs_len = len(obs_token_ids)
                token_stats["total_actual_obs_tokens"] += actual_obs_len
                
                has_img = bool(tool_results.get('image', None))
                has_aud = bool(tool_results.get('audio', None))
                
                if has_img or has_aud:
                    # 此时的 tool_results['obs'] 已经是类似 "System Observation (Compressed): <image>" 的简短文本
                    actual_text_obs = tool_results.get('obs', '')
                    actual_text_len = len(self.tokenizer.encode(actual_text_obs))
                    
                    text_tokens = min(actual_text_len, actual_obs_len)
                    mm_tokens = actual_obs_len - text_tokens
                    
                    if has_img:
                        token_stats["total_obs_image_tokens"] += mm_tokens
                    if has_aud:
                        token_stats["total_obs_audio_tokens"] += mm_tokens
                # =====================================================================

                if 'reward' in tool_results and tool_results['reward'] is not None:
                    stats_dict["rewards"].append(tool_results['reward'] if 'reward' in tool_results else 0.0)
                stats_dict["valid_action"] += tool_results['valid_action'] if 'valid_action' in tool_results else 0
                stats_dict["tool_interact_info"].append(tool_results)
                
                running_prompt_ids.extend(obs_token_ids)
                if self.agent_config.mask_observations:
                    response_mask.extend([0] * len(obs_token_ids))
                else:
                    response_mask.extend([1] * len(obs_token_ids))
                response_logprobs.extend([0.0] * len(obs_token_ids))
                
                if tool_results['done']:
                    stats_dict["is_traj_finished"] = True
                    traj_stop_reason = "tool_signaled_done"
                    break

            # =====================================================================
            # [模型生成 (Model Generation)]
            # =====================================================================
            available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
            max_tokens_for_this_turn = min(max_action_length, available_length)
            
            if max_tokens_for_this_turn <= 0:
                traj_stop_reason = "max_model_len_exceeded"
                break
                
            agent_sampling_params["max_tokens"] = max_tokens_for_this_turn 
            logger.debug(f"Turn {step}: available_length={available_length}, max_tokens_for_this_turn={max_tokens_for_this_turn}")
            
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=running_prompt_ids,
                    sampling_params=agent_sampling_params,
                    image_data=running_image_data,
                    audio_data=running_audio_data,
                )
                if output.text.strip() == "":
                    logger.warning(f"Turn {step}: Generated empty response for traj_id={request_id}. prompt_ids length: {len(running_prompt_ids)}")
                    
            gen_ids = output.token_ids
            gen_logprobs = output.log_probs or [0.0] * len(gen_ids)
            gen_text = output.text

            # ================= [新增模块: 统计模型生成 Token 花费] =================
            token_stats["total_gen_tokens"] += len(gen_ids)
            # =====================================================================

            # =================================================================
            self._append_rollout_jsonl(
                log_filename,
                log_context,
                "model_generation",
                {
                    "turn": step,
                    "text": gen_text,
                    "token_count": len(gen_ids),
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "mean_logprob": float(np.mean(gen_logprobs)) if len(gen_logprobs) > 0 else 0.0,
                },
            )
            # =================================================================

            running_prompt_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            response_logprobs.extend(gen_logprobs)
            
            stats_dict["num_turns"] += 1
            stats_dict["action_lengths"].append(len(gen_ids))
            stats_dict["action_logps"].append(np.mean(gen_logprobs) if len(gen_logprobs) > 0 else 0.0)
            if gen_text.strip() == "":
                stats_dict["empty_responses"] += 1
            
            finish_reason = output.finish_reason
            stop_reason = output.stop_reason
            if isinstance(stop_reason, int):
                stop_reason = self.tokenizer.decode([stop_reason])[0]

            # =====================================================================
            # [动作解析 (Action Parsing)]
            # =====================================================================
            do_action = False
            action_text = ""

            for ext_token in self.action_extract_tokens:
                if ext_token in gen_text:
                    do_action = True
                    last_idx = gen_text.rfind(ext_token)
                    action_text = gen_text[:last_idx + len(ext_token)]
                    break
            
            # =================================================================
            self._append_rollout_jsonl(
                log_filename,
                log_context,
                "action_extract",
                {
                    "turn": step,
                    "do_action": do_action,
                    "action": action_text if action_text.strip() else "",
                    "empty_action": not bool(action_text.strip()),
                },
            )
            # =================================================================

            # =====================================================================
            # [结束条件检查 (End Conditions Check)]
            # =====================================================================
            if not use_tool:
                stats_dict["is_traj_finished"] = True
                traj_stop_reason = f"no_tool-{finish_reason}"
                break
                
            if not do_action or is_last_step:
                if finish_reason == "stop":
                    if not stop_reason:
                        stats_dict["is_traj_finished"] = True
                        if gen_text.strip() == "":
                            traj_stop_reason = "model_chose_to_finish_with_empty_response"
                        else:
                            traj_stop_reason = "model_chose_to_finish"
                    else:
                        stats_dict["is_traj_finished"] = False
                        if do_action and is_last_step:
                            traj_stop_reason = "max_turns_reached"
                        else:
                            traj_stop_reason = "no_action_stop_token_found_in_stop_reason=" + stop_reason
                elif finish_reason == "length":
                    stats_dict["is_traj_finished"] = False
                    traj_stop_reason = "max_response_length_reached"
                else:
                    stats_dict["is_traj_finished"] = True
                    traj_stop_reason = f"finish_reason_{finish_reason}"
                break
                
            # =====================================================================
            # [重新分词 (Retokenization)]
            # =====================================================================
            if self.agent_config.retokenization:
                new_text = self.tokenizer.decode(running_prompt_ids[turn_start_length:], skip_special_tokens=False)
                new_ids = self.tokenizer.encode(new_text)
                if new_ids != running_prompt_ids[turn_start_length:]:
                    stats_dict['retokenization_diff'].append(1)
                    logger.debug(f"Retokenization changed the length from {len(running_prompt_ids) - turn_start_length} to {len(new_ids)}. traj_id={request_id}, turn={step}")
                else:
                    stats_dict['retokenization_diff'].append(0)
                running_prompt_ids = running_prompt_ids[:turn_start_length] + new_ids
                
                mask_len_needed = len(running_prompt_ids) - len(prompt_ids)
                if len(response_mask) > mask_len_needed:
                    response_mask = response_mask[:mask_len_needed]
                else:
                    response_mask.extend([response_mask[-1]] * (mask_len_needed - len(response_mask)))
                
                if len(response_logprobs) > mask_len_needed:
                    response_logprobs = response_logprobs[:mask_len_needed]
                else:
                    response_logprobs.extend([0.0] * (mask_len_needed - len(response_logprobs)))
        
        logger.debug(f"Trajectory {request_id} finished after {step} turns. Stop reason: {traj_stop_reason}")
        response_ids = running_prompt_ids[len(prompt_ids):]
        assert len(response_ids) == len(response_mask), f"Response ids and mask length mismatch: {len(response_ids)} vs {len(response_mask)}"
        
        start = time.time()
        asyncio.create_task(self.close_traj_tool_threads(request_id=request_id))
        end = time.time()
        close_traj_time = end - start
        
        if self.agent_config.mask_overlong_loss and not stats_dict["is_traj_finished"]:
            response_mask = [0] * len(response_mask)
            stats_dict["valid_traj"] = 0
            logger.info(f"Masking the whole response for traj_id={request_id} due to overlong trajectory and not finished.")
        
        if self.agent_config.mask_void_traj:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            has_answer = "\\boxed" in response_text or "final_answer(" in response_text
            if stats_dict["valid_action"] == 0 and not has_answer:
                response_mask = [0] * len(response_mask)
                stats_dict["valid_traj"] = 0
                logger.info(f"Masking the whole response for traj_id={request_id} due to void trajectory (no valid action or no final answer). valid_action={stats_dict['valid_action']}, has_answer={has_answer}")

        if sum(response_mask) == 0:
            if len(response_mask) > 0:
                response_mask[-1] = 1
            else:
                dummy_id = getattr(self.tokenizer, "eos_token_id", 151645) or 151645
                response_ids.append(dummy_id)
                response_mask.append(1)
                response_logprobs.append(0.0)
            
        verl_tool_metrics = {
            "num_turns": stats_dict["num_turns"],
            "empty_responses": stats_dict["empty_responses"],
            "valid_action": stats_dict["valid_action"],
            "per_action_length": np.mean(stats_dict["action_lengths"]) if len(stats_dict["action_lengths"]) > 0 else 0,
            "per_obs_length": np.mean(stats_dict["obs_lengths"]) if len(stats_dict["obs_lengths"]) > 0 else 0,
            "per_action_logp": np.mean(stats_dict["action_logps"]) if len(stats_dict["action_logps"]) > 0 else 0,
            "per_reward_from_tool": np.mean(stats_dict["rewards"]) if len(stats_dict["rewards"]) > 0 else 0,
            "tool_call_success": np.mean([info.get("success", 1.0) for info in stats_dict["tool_interact_info"]]) if len(stats_dict["tool_interact_info"]) > 0 else 1.0,
            "traj_actions_length": sum(stats_dict["action_lengths"]),
            "traj_obs_length": sum(stats_dict["obs_lengths"]),
            "generated_length": len(output.token_ids),
            "is_traj_finished": float(stats_dict["is_traj_finished"]),
            "valid_traj": float(stats_dict["valid_traj"]),
            "no_loss_on_traj": response_mask.count(1) == 0,
            "retokenization_diff": np.mean(stats_dict["retokenization_diff"]) if len(stats_dict["retokenization_diff"]) > 0 else 0,
            "tool_processing_time_sec": sum(info.get("processing_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "tool_queue_time_sec": sum(info.get("queue_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "client_queued_time_sec": sum(info.get("client_queued_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "client_conn_create_time_sec": sum(info.get("client_conn_create_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "client_request_send_time_sec": sum(info.get("client_request_send_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "response_read_time_sec": sum(info.get("response_read_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "response_size_mb": sum(info.get("response_size_mb", 0.0) for info in stats_dict["tool_interact_info"]),
            "response_await_time_sec": sum(info.get("time awaiting response_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "interact_time_time_sec": sum(info.get("interact_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "session_request_time": sum(info.get("session_request_time", 0.0) for info in stats_dict["tool_interact_info"]),
            "X-Process-Time_sec": sum(info.get("X-Process-Time", 0.0) for info in stats_dict["tool_interact_info"]),
            "close_traj_time": close_traj_time,
        }
        
        if stats_dict["tool_interact_info"] and all("metrics" in info for info in stats_dict["tool_interact_info"]):
            tool_metric_keys = stats_dict["tool_interact_info"][0]["metrics"].keys()
            for key in tool_metric_keys:
                try:
                    verl_tool_metrics[f"tool_avg_{key}"] = np.mean([float(info["metrics"][key]) for info in stats_dict["tool_interact_info"] if key in info["metrics"]])
                except Exception as e:
                    logger.warning(f"Failed to compute mean for tool metric {key}: {e}")
                    
        for i, logp in enumerate(stats_dict["action_logps"]):
            verl_tool_metrics[f"turn_{i+1}_action_logp"] = logp
            
        if len(response_ids) > self.response_length:
            cut_index = self.response_length
            last_keep = cut_index - 1
            if last_keep >= 0 and response_ids[last_keep] in self.non_truncate_token_ids:
                while last_keep >= 0 and response_ids[last_keep] in self.non_truncate_token_ids:
                    last_keep -= 1
                cut_index = last_keep + 1
            response_ids = response_ids[:cut_index]
            response_mask = response_mask[:cut_index]
            response_logprobs = response_logprobs[:cut_index]

        if running_image_data is not None:
            full_ids = prompt_ids + response_ids
            vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            num_visual_segments = full_ids.count(vision_start_id)
            if len(running_image_data) > num_visual_segments:
                running_image_data = running_image_data[:num_visual_segments]
        
        multi_modal_output = {}
        if running_image_data is not None:
            multi_modal_output["image"] = running_image_data
        if running_audio_data is not None:
            multi_modal_output["audio"] = running_audio_data

        # ================= [新增模块: 在轨迹日志末尾输出 Token 统计] =================
        total_actual_tokens = token_stats["initial_prompt_tokens"] + token_stats["total_actual_obs_tokens"] + token_stats["total_gen_tokens"]
        
        # 计算压缩收益
        original_cost = token_stats["total_original_obs_text_tokens"]
        actual_cost = token_stats["total_actual_obs_tokens"]
        tokens_saved = original_cost - actual_cost
        compression_ratio = (original_cost / actual_cost) if actual_cost > 0 else 1.0

        # self._append_rollout_jsonl(
        #     log_filename,
        #     log_context,
        #     "trajectory_summary",
        #     {
        #         "stop_reason": traj_stop_reason,
        #         "is_traj_finished": stats_dict["is_traj_finished"],
        #         "valid_traj": stats_dict["valid_traj"],
        #         "num_turns": stats_dict["num_turns"],
        #         "token_usage": {
        #             "initial_prompt": token_stats["initial_prompt_tokens"],
        #             "model_generation": token_stats["total_gen_tokens"],
        #             "obs_original_text": original_cost,
        #             "obs_actual_fed": actual_cost,
        #             "obs_image": token_stats["total_obs_image_tokens"],
        #             "obs_audio": token_stats["total_obs_audio_tokens"],
        #             "saved_by_compression": tokens_saved,
        #             "compression_ratio": compression_ratio,
        #             "grand_total_fed": total_actual_tokens,
        #         },
        #         "verl_tool_metrics": verl_tool_metrics,
        #     },
        # )
            
        # 同步更新 verl_tool_metrics (供 Wandb 监控大盘使用)
        verl_tool_metrics.update({
            "token_initial_prompt": token_stats["initial_prompt_tokens"],
            "token_total_gen": token_stats["total_gen_tokens"],
            "token_obs_original_text": original_cost,
            "token_obs_actual_fed": actual_cost,
            "token_obs_image": token_stats["total_obs_image_tokens"],
            "token_saved_by_compression": tokens_saved,
            "token_grand_total": total_actual_tokens,
        })
        # =====================================================================

        tool_interact_info = stats_dict.get("tool_interact_info", [])
        if self.agent_config.compact_tool_interact_info:
            tool_interact_info = compact_tool_interact_info_entries(tool_interact_info)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length],
            multi_modal_data=multi_modal_output,
            num_turns=stats_dict["num_turns"],
            metrics=metrics,
            extra_fields={"tool_interact_info": tool_interact_info, "traj_stop_reason": traj_stop_reason, "verl_tool_metrics": verl_tool_metrics},
        )
        return output
