import asyncio
import json
import multiprocessing as mp
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import ray

from .base import BaseTool, register_tool


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG = _env_flag("TEXT_BROWSER_DEBUG", False)
ENABLE_BATCH_LOG = _env_flag("TEXT_BROWSER_BATCH_LOG", False)
ACTION_PATTERN = re.compile(
    r"<think>.*?</think>\s*(?:```.*?```|<action>.*?</action>)",
    re.DOTALL,
)
RAW_COMMAND_PATTERN = re.compile(
    r"^\s*(?:```)?\s*(click|type|hover|press|scroll|new_tab|tab_focus|close_tab|goto|go_back|go_forward|stop)\b",
    re.IGNORECASE | re.DOTALL,
)
XML_ACTION_PATTERN = re.compile(
    r"^\s*<\s*(click|type|hover|press|scroll|new_tab|tab_focus|close_tab|goto|go_back|go_forward|stop)\b.*?>",
    re.IGNORECASE | re.DOTALL,
)
MAX_ACTIVE_ACTORS = int(os.getenv("TEXT_BROWSER_MAX_ACTIVE_ACTORS", "512"))
IDLE_ACTOR_POOL_SIZE = int(os.getenv("TEXT_BROWSER_IDLE_POOL_SIZE", "64"))
ACTOR_TTL_SECONDS = int(os.getenv("TEXT_BROWSER_ACTOR_TTL_SECONDS", "1200"))
ACTOR_TTL_POLL_SECONDS = int(os.getenv("TEXT_BROWSER_ACTOR_TTL_POLL_SECONDS", "120"))
ACTOR_CPU_FRACTION = float(os.getenv("TEXT_BROWSER_ACTOR_CPUS", "0.25"))
INIT_RETRIES = max(1, int(os.getenv("TEXT_BROWSER_INIT_RETRIES", "1")))
STEP_RETRIES = max(1, int(os.getenv("TEXT_BROWSER_STEP_RETRIES", "2")))
INIT_RETRY_BACKOFF_SEC = float(os.getenv("TEXT_BROWSER_INIT_RETRY_BACKOFF_SEC", "1.0"))
STEP_RETRY_BACKOFF_SEC = float(os.getenv("TEXT_BROWSER_STEP_RETRY_BACKOFF_SEC", "0.5"))
ENV_PROCESS_RPC_TIMEOUT_SEC = float(os.getenv("TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC", "300.0"))
ENV_PROCESS_SHUTDOWN_TIMEOUT_SEC = float(
    os.getenv("TEXT_BROWSER_ENV_SHUTDOWN_TIMEOUT_SEC", "10.0")
)


def _debug(message: str) -> None:
    if DEBUG:
        print(f"[TEXT_BROWSER] {message}")


def _env_process_main(conn) -> None:
    env = None
    try:
        from mini_webarena.env_worker import WikiQAEnv

        while True:
            try:
                request = conn.recv()
            except EOFError:
                break

            cmd = request.get("cmd")
            payload = request.get("payload", {})

            try:
                if cmd == "ping":
                    result = True
                elif cmd == "create":
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    env = WikiQAEnv(
                        payload["question"],
                        payload["gt"],
                        url=payload.get("url"),
                        prompt_format="last",
                    )
                    result = env.render()
                elif cmd == "render":
                    if env is None:
                        raise RuntimeError("WikiQAEnv is not initialized.")
                    result = env.render()
                elif cmd == "step":
                    if env is None:
                        raise RuntimeError("WikiQAEnv is not initialized.")
                    result = env.step(payload["action"])
                elif cmd == "close_env":
                    if env is not None:
                        try:
                            env.close()
                        finally:
                            env = None
                    result = True
                elif cmd == "shutdown":
                    if env is not None:
                        try:
                            env.close()
                        finally:
                            env = None
                    conn.send({"status": "ok", "result": True})
                    break
                else:
                    raise ValueError(f"Unknown env process command: {cmd}")

                conn.send({"status": "ok", "result": result})
            except Exception as exc:
                conn.send(
                    {
                        "status": "error",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class _EnvProcessClient:
    def __init__(self):
        start_method = os.getenv("TEXT_BROWSER_MP_START_METHOD", "spawn")
        self._ctx = mp.get_context(start_method)
        self._conn = None
        self._process = None
        self._lock = threading.RLock()

    def _ensure_started_locked(self) -> None:
        if self._process is not None and self._process.is_alive() and self._conn is not None:
            return

        self._cleanup_transport_locked()
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_env_process_main,
            args=(child_conn,),
            name="wiki_env_process",
        )
        process.start()
        child_conn.close()
        self._conn = parent_conn
        self._process = process

    def _cleanup_transport_locked(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._process = None

    def _terminate_process_locked(self) -> None:
        process = self._process
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=ENV_PROCESS_SHUTDOWN_TIMEOUT_SEC)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=ENV_PROCESS_SHUTDOWN_TIMEOUT_SEC)
            except Exception as exc:
                _debug(f"env process terminate failed: {exc}")
        self._cleanup_transport_locked()

    def _raise_remote_error(self, response: Dict[str, Any]) -> None:
        error_type = response.get("error_type", "RuntimeError")
        error_message = response.get("error", "Unknown env process error")
        tb = response.get("traceback")
        if tb:
            raise RuntimeError(f"{error_type}: {error_message}\n{tb}")
        raise RuntimeError(f"{error_type}: {error_message}")

    def call(self, cmd: str, timeout: float = ENV_PROCESS_RPC_TIMEOUT_SEC, **payload):
        with self._lock:
            self._ensure_started_locked()

            try:
                self._conn.send({"cmd": cmd, "payload": payload})
            except Exception as exc:
                self._terminate_process_locked()
                raise RuntimeError(f"Failed to send env process command '{cmd}': {exc}") from exc

            if not self._conn.poll(timeout):
                self._terminate_process_locked()
                raise TimeoutError(
                    f"Timed out waiting for env process command '{cmd}' after {timeout}s"
                )

            try:
                response = self._conn.recv()
            except EOFError as exc:
                self._terminate_process_locked()
                raise RuntimeError(
                    f"Env process exited unexpectedly while handling '{cmd}'"
                ) from exc

            if response.get("status") == "ok":
                return response.get("result")

            self._raise_remote_error(response)

    def close_env(self) -> bool:
        with self._lock:
            if self._process is None:
                return True
        return bool(self.call("close_env", timeout=ENV_PROCESS_SHUTDOWN_TIMEOUT_SEC))

    def shutdown(self) -> bool:
        with self._lock:
            if self._process is None:
                return True

            if self._process.is_alive() and self._conn is not None:
                try:
                    self._conn.send({"cmd": "shutdown", "payload": {}})
                    if self._conn.poll(ENV_PROCESS_SHUTDOWN_TIMEOUT_SEC):
                        self._conn.recv()
                except Exception as exc:
                    _debug(f"env process shutdown RPC failed: {exc}")

            self._terminate_process_locked()
            return True


@ray.remote(max_restarts=0)
class WikiEnvActor:
    def __init__(self):
        self._env_runtime = _EnvProcessClient()
        self._env_started = False
        self._initial_observation = None
        self._last_access = time.time()
        self._ttl_seconds = ACTOR_TTL_SECONDS
        self._state_lock = threading.Lock()
        self._active_requests = 0

        def _watchdog():
            while True:
                time.sleep(ACTOR_TTL_POLL_SECONDS)
                with self._state_lock:
                    idle_for = time.time() - self._last_access
                    active_requests = self._active_requests
                if idle_for <= self._ttl_seconds or active_requests > 0:
                    continue
                _debug("WikiEnvActor idle timeout reached, closing actor.")
                try:
                    self.shutdown()
                    ray.actor.exit_actor()
                except Exception:
                    import os as _os

                    _os._exit(0)

        threading.Thread(target=_watchdog, daemon=True).start()

    def _begin_request(self) -> None:
        with self._state_lock:
            self._active_requests += 1
            self._last_access = time.time()

    def _end_request(self) -> None:
        with self._state_lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_access = time.time()

    def _close_env(self) -> None:
        try:
            self._env_runtime.close_env()
        finally:
            self._env_started = False
            self._initial_observation = None

    def _create_env(self, question: str, gt: str, url: str = None) -> None:
        last_error = None
        for attempt in range(INIT_RETRIES):
            try:
                self._initial_observation = self._env_runtime.call(
                    "create",
                    question=question,
                    gt=gt,
                    url=url,
                )
                self._env_started = True
                return
            except Exception as exc:
                last_error = exc
                self._env_started = False
                self._env_runtime.shutdown()
                if attempt + 1 < INIT_RETRIES:
                    time.sleep(INIT_RETRY_BACKOFF_SEC * (attempt + 1))
        raise last_error

    def _render_env(self) -> str:
        return self._env_runtime.call("render")

    def _step_env(self, action: str):
        return self._env_runtime.call("step", action=action)

    def reset_env(self) -> bool:
        self._begin_request()
        try:
            self._close_env()
            return True
        finally:
            self._end_request()

    def shutdown(self) -> bool:
        self._begin_request()
        try:
            self._env_started = False
            self._initial_observation = None
            self._env_runtime.shutdown()
            return True
        finally:
            self._end_request()

    def ping(self) -> bool:
        return True

    def execute(self, action: str, question: str, gt: str, url: str = None):
        self._begin_request()
        try:
            if not self._env_started:
                self._create_env(question, gt, url)

            if action is None or action == "":
                if self._initial_observation is None:
                    self._initial_observation = self._render_env()
                return self._initial_observation, False, True

            last_error = None
            for attempt in range(STEP_RETRIES):
                try:
                    obs, done, valid = self._step_env(action)
                    if done:
                        self._close_env()
                    else:
                        self._initial_observation = None
                    return obs, done, valid
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < STEP_RETRIES:
                        time.sleep(STEP_RETRY_BACKOFF_SEC * (attempt + 1))
            raise last_error
        finally:
            self._end_request()


@register_tool
class TextBrowserTool(BaseTool):
    """
    TextBrowserTool uses Ray actors to manage WikiQAEnv sessions.
    Each trajectory_id has a dedicated actor. It supports initial
    render (action=None) and step operations.
    """

    tool_type = "text_browser"

    def __init__(self, num_workers=32):
        super().__init__(num_workers)
        self.env_actors = {}
        self.actor_creation_order = []
        self.idle_actors = []
        self.actor_num_cpus = ACTOR_CPU_FRACTION
        self._lock = threading.RLock()

    def get_usage_inst(self) -> str:
        return "TextBrowserTool uses Ray actors to manage WikiQAEnv sessions."

    def has_env(self, trajectory_id):
        with self._lock:
            return trajectory_id in self.env_actors

    def load_env(self, trajectory_id: str):
        with self._lock:
            return self.env_actors.get(trajectory_id)

    def _new_actor(self):
        return WikiEnvActor.options(num_cpus=self.actor_num_cpus).remote()

    def _acquire_actor(self):
        while True:
            with self._lock:
                actor = self.idle_actors.pop() if self.idle_actors else None
            if actor is None:
                break
            try:
                ray.get(actor.ping.remote(), timeout=2)
                _debug("reusing idle WikiEnvActor")
                return actor
            except Exception as exc:
                _debug(f"dropping dead idle WikiEnvActor: {exc}")
        _debug("creating new WikiEnvActor")
        return self._new_actor()

    def _discard_actor(self, actor):
        try:
            try:
                ray.get(actor.shutdown.remote(), timeout=5)
            except Exception as exc:
                _debug(f"actor.shutdown failed before discard: {exc}")
            ray.kill(actor, no_restart=True)
        except Exception as exc:
            _debug(f"ray.kill failed while discarding actor: {exc}")

    def _release_actor(self, actor):
        kill_actor = False
        with self._lock:
            if len(self.idle_actors) < IDLE_ACTOR_POOL_SIZE:
                self.idle_actors.append(actor)
            else:
                kill_actor = True
        if kill_actor:
            self._discard_actor(actor)

    def save_env(self, trajectory_id: str, actor):
        with self._lock:
            existing = self.env_actors.get(trajectory_id)
            if existing is None:
                self.env_actors[trajectory_id] = actor
                self.actor_creation_order.append(trajectory_id)
            else:
                if existing != actor:
                    raise RuntimeError(
                        f"Actor with trajectory_id {trajectory_id} already exists."
                    )
                if trajectory_id in self.actor_creation_order:
                    self.actor_creation_order.remove(trajectory_id)
                self.actor_creation_order.append(trajectory_id)
        self._cleanup_actors_if_needed()

    async def asave_env(self, trajectory_id: str, actor):
        self.save_env(trajectory_id, actor)

    def _pop_actor(self, trajectory_id):
        with self._lock:
            actor = self.env_actors.pop(trajectory_id, None)
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            return actor

    def delete_env(self, trajectory_id, reset_actor=False):
        actor = self._pop_actor(trajectory_id)
        if actor is None:
            return
        release_actor = True
        if reset_actor:
            try:
                ray.get(actor.reset_env.remote())
            except Exception as exc:
                _debug(f"actor.reset_env failed for {trajectory_id}: {exc}")
                release_actor = False
        if release_actor:
            self._release_actor(actor)
        else:
            self._discard_actor(actor)

    async def adelete_env(self, trajectory_id, reset_actor=False):
        actor = self._pop_actor(trajectory_id)
        if actor is None:
            return
        release_actor = True
        if reset_actor:
            try:
                await actor.reset_env.remote()
            except Exception as exc:
                _debug(f"actor.reset_env failed for {trajectory_id}: {exc}")
                release_actor = False
        if release_actor:
            self._release_actor(actor)
        else:
            self._discard_actor(actor)

    def parse_action(self, action):
        if action in ("", None):
            return action, True
        if ACTION_PATTERN.search(action):
            return action, True
        if RAW_COMMAND_PATTERN.search(action):
            return action, True
        if XML_ACTION_PATTERN.search(action):
            return action, True
        return action, False

    def _touch_actor(self, trajectory_id: str) -> None:
        with self._lock:
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            self.actor_creation_order.append(trajectory_id)

    @staticmethod
    def _extract_request_context(extra_field: dict):
        return (
            extra_field.get("question", "placeholder"),
            extra_field.get("gt", "placeholder"),
            extra_field.get("url"),
        )

    def _run_actor(self, actor, action: str, extra_field: dict):
        question, gt, url = self._extract_request_context(extra_field)
        return ray.get(actor.execute.remote(action, question, gt, url))

    async def _arun_actor(self, actor, action: str, extra_field: dict):
        question, gt, url = self._extract_request_context(extra_field)
        return await asyncio.wait_for(
            actor.execute.remote(action, question, gt, url),
            timeout=300,
        )

    def get_current_observation(self, trajectory_id: str, extra_field: dict):
        actor = self.load_env(trajectory_id)
        if actor is None:
            return None

        try:
            obs, _, _ = self._run_actor(actor, "", extra_field)
        except Exception as exc:
            _debug(f"get_current_observation failed for {trajectory_id}: {exc}")
            return None

        self._touch_actor(trajectory_id)
        return obs

    async def aget_current_observation(self, trajectory_id: str, extra_field: dict):
        actor = self.load_env(trajectory_id)
        if actor is None:
            return None

        try:
            obs, _, _ = await self._arun_actor(actor, "", extra_field)
        except Exception as exc:
            _debug(f"aget_current_observation failed for {trajectory_id}: {exc}")
            return None

        self._touch_actor(trajectory_id)
        return obs

    def conduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        actor = self.load_env(trajectory_id)
        if actor is None:
            actor = self._acquire_actor()
            self.save_env(trajectory_id, actor)

        try:
            obs, done, valid = self._run_actor(actor, action, extra_field)
        except Exception as exc:
            self.delete_env(trajectory_id, reset_actor=True)
            return f"Error: {exc}", False, False

        self._touch_actor(trajectory_id)

        if not valid:
            obs = (
                "The action is invalid, please retry\n\n"
                f"Current observation:\n{obs}"
            )

        if done:
            if valid:
                obs = ""
            self.delete_env(trajectory_id, reset_actor=False)

        return obs, done, valid

    async def aconduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        actor = self.load_env(trajectory_id)
        if actor is None:
            actor = self._acquire_actor()
            await self.asave_env(trajectory_id, actor)

        try:
            obs, done, valid = await self._arun_actor(actor, action, extra_field)
        except asyncio.TimeoutError:
            await self.adelete_env(trajectory_id, reset_actor=True)
            return "[TIMEOUT] (aconduct_action)", True, False
        except Exception as exc:
            await self.adelete_env(trajectory_id, reset_actor=True)
            return f"Error: {exc}", False, False

        self._touch_actor(trajectory_id)
        obs_before_cleanup = obs

        if done:
            if valid:
                obs = ""
            await self.adelete_env(trajectory_id, reset_actor=False)

        if not valid:
            obs = (
                "The action is invalid, please retry\n\n"
                f"Current observation:\n{obs_before_cleanup}"
            )

        return obs, done, valid

    def _maybe_log_batch(self, trajectory_ids, actions, extra_fields, observations, dones, valid_flags):
        if not ENABLE_BATCH_LOG:
            return
        try:
            from pathlib import Path

            log_path = Path("browser_server_logs.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "input": {
                                "trajectory_ids": trajectory_ids,
                                "actions": actions,
                                "extra_fields": extra_fields,
                            },
                            "output": {
                                "observations": observations,
                                "dones": dones,
                                "valid_flags": valid_flags,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            _debug(f"batch log write failed: {exc}")

    async def aget_observations(self, trajectory_ids, actions, extra_fields):
        sem = asyncio.Semaphore(self.num_workers)

        async def _task(index: int):
            async with sem:
                extra = extra_fields[index].get("extra_fields", extra_fields[index])
                return index, *await self.aconduct_action(
                    trajectory_ids[index], actions[index], extra
                )

        results = await asyncio.gather(*[_task(i) for i in range(len(trajectory_ids))])

        observations = [""] * len(trajectory_ids)
        dones = [False] * len(trajectory_ids)
        valid_flags = [True] * len(trajectory_ids)

        for index, obs, done, valid in results:
            observations[index] = obs
            dones[index] = done
            valid_flags[index] = valid

        self._maybe_log_batch(
            trajectory_ids, actions, extra_fields, observations, dones, valid_flags
        )
        return observations, dones, valid_flags

    def get_observations(self, trajectory_ids, actions, extra_fields):
        total = len(trajectory_ids)
        observations = [""] * total
        dones = [False] * total
        valid_flags = [True] * total

        def _worker(index: int):
            extra = extra_fields[index].get("extra_fields", extra_fields[index])
            return index, *self.conduct_action(trajectory_ids[index], actions[index], extra)

        if total <= 1:
            results = [_worker(0)] if total == 1 else []
        else:
            with ThreadPoolExecutor(max_workers=min(self.num_workers, total)) as pool:
                results = list(pool.map(_worker, range(total)))

        for index, obs, done, valid in results:
            observations[index] = obs
            dones[index] = done
            valid_flags[index] = valid

        self._maybe_log_batch(
            trajectory_ids, actions, extra_fields, observations, dones, valid_flags
        )
        return observations, dones, valid_flags

    def _cleanup_actors_if_needed(self):
        victims = []
        with self._lock:
            while len(self.env_actors) > MAX_ACTIVE_ACTORS and self.actor_creation_order:
                oldest = self.actor_creation_order.pop(0)
                actor = self.env_actors.pop(oldest, None)
                if actor is not None:
                    victims.append(actor)
        for actor in victims:
            self._discard_actor(actor)
