"""RunManager: manages executor lifecycle and run execution in a background thread."""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Dict, Optional

from config import Config
from web.observer import TreeObserver
from web.web_config import build_web_config


class RunStatus(str, Enum):
    IDLE = "idle"
    LOADING_MODEL = "loading_model"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class RunState:
    run_id: str
    status: RunStatus
    config: Config
    event_queue: Queue = field(default_factory=Queue)
    thread: Optional[threading.Thread] = None
    cancel_requested: bool = False
    error: Optional[str] = None


class RunManager:
    """Singleton that manages one BOA run at a time."""

    def __init__(self):
        self.current_run: Optional[RunState] = None
        self._executor = None
        self._cached_model: Optional[str] = None

    @property
    def status(self) -> RunStatus:
        if self.current_run is None:
            return RunStatus.IDLE
        return self.current_run.status

    def start_run(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        top_k: int,
        likelihood: float,
        search_strategy: str = "greedy",
        pa_min_depth: int = 3,
        pa_shallow_samples: int = 20,
        pa_shallow_score: float = 8000,
        pa_subtree_top_k: int = 5,
        pa_search_temperature: float = 0.5,
        mcts_exploration_constant: float = 1.414,
        judger_type: str = "nuanced",
        custom_judger_prompt: str = "",
        time_limit_sec: float = 120,
        chunk_size: int = 2,
    ) -> str:
        """Start a new run. Returns the run_id."""
        if self.current_run and self.current_run.status == RunStatus.RUNNING:
            raise RuntimeError("A run is already in progress. Cancel it first.")

        # Wait for a cancelling run's thread to finish before starting a new one
        if self.current_run and self.current_run.status == RunStatus.CANCELLING:
            thread = self.current_run.thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("Previous run is still cancelling. Please try again shortly.")

        cfg = build_web_config(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            likelihood=likelihood,
            search_strategy=search_strategy,
            pa_min_depth=pa_min_depth,
            pa_shallow_samples=pa_shallow_samples,
            pa_shallow_score=pa_shallow_score,
            pa_subtree_top_k=pa_subtree_top_k,
            pa_search_temperature=pa_search_temperature,
            mcts_exploration_constant=mcts_exploration_constant,
            judger_type=judger_type,
            custom_judger_prompt=custom_judger_prompt,
            time_limit_sec=time_limit_sec,
            chunk_size=chunk_size,
        )

        run_id = f"web_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        state = RunState(run_id=run_id, status=RunStatus.LOADING_MODEL, config=cfg)

        self.current_run = state

        thread = threading.Thread(
            target=self._run_thread,
            args=(state, prompt),
            daemon=True,
            name=f"boa-web-{run_id}",
        )
        state.thread = thread
        thread.start()

        return run_id

    def cancel_run(self, run_id: str) -> bool:
        """Request cancellation of a run."""
        if self.current_run and self.current_run.run_id == run_id:
            self.current_run.cancel_requested = True
            self.current_run.status = RunStatus.CANCELLING
            # Set a very short time limit to make the executor stop quickly
            if self._executor and hasattr(self._executor, 'runtime_guard'):
                self._executor.runtime_guard.time_limit_sec = 0.0
            return True
        return False

    def sample_sequences(self, n: int = 5) -> list:
        """Sample N complete responses from the root node of the last run.

        Returns list of {text, cum_log_prob, probability}.
        """
        if self._executor is None:
            raise RuntimeError("No executor available. Run a prompt first.")
        if self.current_run is None:
            raise RuntimeError("No run available.")
        if getattr(self._executor, "root", None) is None:
            raise RuntimeError("No root node from previous run.")
        return self._executor._sample_from_root(n=n)

    def _free_executor(self) -> None:
        """Release old executor and free GPU memory."""
        import gc
        import torch

        executor = self._executor
        self._executor = None
        self._cached_model = None

        # Delete engine references to release GPU tensors
        if executor is not None:
            for attr in ("target_model_engine", "judger_engine", "local_judger_engine",
                         "api_judger_engine", "expander", "sampler", "sample_worker",
                         "judger", "judge_worker"):
                try:
                    setattr(executor, attr, None)
                except Exception:
                    pass
            del executor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _run_thread(self, state: RunState, prompt: str) -> None:
        """Background thread that runs the BOA executor."""
        observer = TreeObserver(state.event_queue)

        try:
            cfg = state.config

            # Set up logging (same as run.py)
            from utils.logger import set_default_log_file, set_default_log_mode
            from utils.run_naming import build_log_file_path
            cfg.run_id = state.run_id  # use the run_id already assigned by start_run
            log_file = build_log_file_path(cfg, run_id=state.run_id)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            set_default_log_file(log_file)
            set_default_log_mode(cfg.logger_mode)

            # Send status update
            state.event_queue.put({"type": "status", "data": {"status": "loading_model"}})

            # Check if we can reuse the cached executor
            need_reload = (
                self._executor is None
                or self._cached_model != cfg.target_model
            )

            if need_reload:
                # Free old model from GPU before loading new one
                if self._executor is not None:
                    self._free_executor()

                from executor.executor import Executor
                self._executor = Executor(
                    cfg,
                    threshold=None,
                    init_target=True,
                    init_judger=True,
                )
                self._cached_model = cfg.target_model
            else:
                # Reuse existing executor with new config and reporter
                old_workload = self._executor.config.workload_name
                self._executor.update_config(cfg)
                from reporters.reporter import Reporter
                self._executor.reporter = Reporter(cfg)
                # Reset routing policy for fresh judgment summary
                from executor.routing_policy import RoutingPolicy
                self._executor.routing_policy = RoutingPolicy(cfg)

                # Re-initialize judger if workload changed (different judger model/profile)
                if old_workload != cfg.workload_name:
                    self._executor._judger_initialized = False
                    self._executor.initialize_judger_components()

            executor = self._executor
            # Clear any leftover observers from previous runs
            executor._observers.clear()
            executor.register_observer(observer)

            state.status = RunStatus.RUNNING
            state.event_queue.put({"type": "status", "data": {"status": "running"}})

            # Build the sample dict as run.py does
            sample = {
                "prompt": prompt,
                "original_prompt": prompt,
            }
            if cfg.workload_name == "agent":
                sample["environments"] = []
                sample["tool_schemas"] = []
                sample["tools_openai"] = []
                sample["fulfillable"] = True

            prompt_metadata = {
                "environments": sample.get("environments", []),
                "tool_schemas": sample.get("tool_schemas", []),
                "tools_openai": sample.get("tools_openai", []),
                "fulfillable": sample.get("fulfillable"),
            }

            executor.run(
                prompt,
                prompt,  # original_prompt = prompt
                prompt_metadata=prompt_metadata,
                prompt_index=1,
                total_prompts=1,
            )

            state.status = RunStatus.COMPLETED

        except Exception as e:
            state.status = RunStatus.ERROR
            state.error = str(e)
            observer.on_error(e)

        finally:
            # Remove observer to avoid leaks
            if self._executor and observer in self._executor._observers:
                self._executor._observers.remove(observer)


# Singleton
run_manager = RunManager()
