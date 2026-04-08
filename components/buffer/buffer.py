from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import threading
from boa_types.tree_node import TreeNode
from utils.conversation_formatter import build_node_model_input

@dataclass
class BufferItem:
    """
    One sampling/judging task routed by Executor.

    Field semantics:
    - `path_text` / `path_ids`:
      Full context fed to the LLM for this sampling request.
      Built on demand from the node's conversation state or legacy tree path.
    - `seq_text` / `seq_ids`:
      Generated sample for THIS sampling request.
      `seq_text` is incremental generated text only (does not include prompt/path_text).
      `seq_ids` is currently stored as full ids (`path_ids + generated_ids`) by Executor.
      `seq_new_ids` stores incremental generated token ids only.
    - `judger_prompt`:
      Original prompt payload for the judger. For agent workloads this stays as the structured JSON prompt.
    - `judger_metadata`:
      Extra metadata needed by the judger, such as tool schemas and environment info.
    - `original_prompt`:
      Raw user request without chat template. Kept for logging and reporting.
    - `node`:
      Back-reference to source TreeNode for appending scores / queue routing.
    """
    node: Any
    path_text: str
    path_ids: List[int]
    base_generated_len: int = 0
    judger_prompt: Optional[str] = None
    judger_metadata: Dict[str, Any] | None = None
    original_prompt: Optional[str] = None
    # Snapshot of the node's assistant text at enqueue time.
    # This is the immutable base for response reconstruction — JudgeWorker
    # concatenates this with seq_text instead of reading live node state.
    node_assistant_text: str = ""
    seq_text: Optional[str] = None   # Filled by Sampler
    seq_ids: Optional[List[int]] = None
    seq_new_ids: Optional[List[int]] = None
    is_complete: bool = False         # True if generation ended with EOS (not max-length truncated)
    score: Optional[float] = None
    response_tau: Optional[float] = None  # Cumulative log-prob of generated tokens (from sampler tau)

class Buffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        # Deque for efficient FIFO
        self._queue: deque[BufferItem] = deque()
        self._lock = threading.Lock()

    # --- Sample Buffer: create N items per node ---
    def add_requests(
        self,
        node: TreeNode,
        num_needed: int,
        tokenizer: Any,
        model_name: str = "",
        judger_prompt: Optional[str] = None,
        judger_metadata: Optional[Dict[str, Any]] = None,
        original_prompt: Optional[str] = None,
    ) -> bool:
        """
        Expand one node into N BufferItems.
        `path_*` is full path context (not incremental chunk); see CONVENTIONS.md.
        """
        to_add = max(0, int(num_needed or 0))
        if to_add <= 0:
            return len(self) >= self.capacity
        path_text, path_ids, base_generated_len = build_node_model_input(
            node,
            tokenizer,
            model_name=model_name,
        )
        # Snapshot assistant text now so judger reads a frozen value,
        # not live node state that could theoretically change later.
        assistant_text_snapshot = node.conversation_state.latest_assistant_text()
        with self._lock:
            for _ in range(to_add):
                item = BufferItem(
                    node=node,
                    path_text=path_text,
                    path_ids=path_ids,
                    base_generated_len=base_generated_len,
                    node_assistant_text=assistant_text_snapshot,
                    judger_prompt=judger_prompt,
                    judger_metadata=dict(judger_metadata or {}),
                    original_prompt=original_prompt,
                )
                self._queue.append(item)
            return len(self._queue) >= self.capacity

    # --- Judger Buffer: append existing item ---
    def add_item(self, item: BufferItem) -> bool:
        """Append an existing item (e.g. from Sample Buffer)."""
        with self._lock:
            self._queue.append(item)
            return len(self._queue) >= self.capacity



    def pop_batch(self, count: int) -> List[BufferItem]:
        """Pop up to `count` items in FIFO order."""
        with self._lock:
            if not self._queue:
                return []

            n = min(int(count), len(self._queue))
            if n == len(self._queue):
                items: List[BufferItem] = list(self._queue)
                self._queue.clear()
                return items
            return [self._queue.popleft() for _ in range(n)]

    def pop_all(self) -> List[BufferItem]:
        """Pop all items in FIFO order."""
        with self._lock:
            if not self._queue:
                return []
            items: List[BufferItem] = list(self._queue)
            self._queue.clear()
            return items

    def is_full(self) -> bool:
        with self._lock:
            return len(self._queue) >= self.capacity

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0
    
    def __len__(self):
        with self._lock:
            return len(self._queue)
