"""
ProfileSession: thread-safe per-run timing accumulator.

Tracks inclusive time (cumulative, including sub-calls) and exclusive / self
time (subtracting time spent in nested profiled calls) for each named block.
Also records the caller→callee edges so the output can be rendered as a tree.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class FuncStats:
    calls: int = 0
    total_sec: float = 0.0   # inclusive: wall time including sub-calls
    self_sec: float = 0.0    # exclusive: wall time minus time in nested profiled calls
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def update(self, elapsed: float, child_sec: float) -> None:
        self.calls += 1
        self.total_sec += elapsed
        self.self_sec += max(0.0, elapsed - child_sec)
        self.min_sec = min(self.min_sec, elapsed)
        self.max_sec = max(self.max_sec, elapsed)

    @property
    def mean_sec(self) -> float:
        return self.total_sec / self.calls if self.calls > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "total_sec": round(self.total_sec, 4),
            "self_sec": round(self.self_sec, 4),
            "mean_sec": round(self.mean_sec, 4),
            "min_sec": round(self.min_sec, 4) if self.min_sec != float("inf") else None,
            "max_sec": round(self.max_sec, 4),
        }


class ProfileSession:
    """
    A named profiling session. Create one per run, then wrap operations with
    @profile or profile_block.

    Thread-safe: accumulation uses a lock; call-stack tracking is thread-local
    so nested calls within one thread correctly compute exclusive (self) time.
    Also records caller→callee edges to enable tree-shaped output.
    """

    def __init__(self, name: str = "session", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._lock = threading.Lock()
        self._stats: Dict[str, FuncStats] = {}
        self._edges: Dict[str, Set[str]] = {}   # parent -> set of children
        self._local = threading.local()          # per-thread call stack

    # ------------------------------------------------------------------
    # Internal stack helpers (called by decorators/context managers)
    # ------------------------------------------------------------------

    def _enter(self, name: str) -> float:
        """Record entry into a profiled region. Returns start timestamp."""
        t = time.perf_counter()
        stack: List[Tuple[str, float, float]] = self._get_stack()
        # (region_name, start_time, accumulated_child_time)
        stack.append((name, t, 0.0))
        return t

    def _exit(self, name: str, start: float) -> None:
        """Record exit from a profiled region and update stats."""
        elapsed = time.perf_counter() - start
        stack = self._get_stack()

        # Pop our own frame (should be the top of the stack)
        child_sec = 0.0
        if stack and stack[-1][0] == name:
            _, _, child_sec = stack.pop()
        else:
            # Mismatched exit (shouldn't happen in normal usage); pop anyway
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    _, _, child_sec = stack.pop(i)
                    break

        # Credit our elapsed time to the parent frame's child_sec
        # and record the caller→callee edge
        if stack:
            pname, pstart, pchild = stack[-1]
            stack[-1] = (pname, pstart, pchild + elapsed)
            with self._lock:
                self._edges.setdefault(pname, set()).add(name)

        # Accumulate into stats
        with self._lock:
            if name not in self._stats:
                self._stats[name] = FuncStats()
            self._stats[name].update(elapsed, child_sec)

    def _get_stack(self) -> List[Tuple[str, float, float]]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _roots(self) -> List[str]:
        """Names that are never a child of another profiled function."""
        with self._lock:
            all_children: Set[str] = set()
            for children in self._edges.values():
                all_children |= children
            roots = [n for n in self._stats if n not in all_children]
        # Sort roots by total_sec descending so the hottest path comes first
        roots.sort(key=lambda n: -self._stats[n].total_sec)
        return roots

    def _tree_lines(self) -> List[str]:
        """Return lines for the tree section of the text report."""
        with self._lock:
            edges = {k: sorted(v, key=lambda n: -self._stats[n].total_sec)
                     for k, v in self._edges.items()}
            stats = {k: v.to_dict() for k, v in self._stats.items()}

        roots = self._roots()
        col = 44
        total_wall = sum(stats[r]["total_sec"] for r in roots) or 1.0

        def _row(name: str, prefix: str, is_last: bool) -> List[str]:
            connector = "└── " if is_last else "├── "
            s = stats[name]
            min_s = s["min_sec"] if s["min_sec"] is not None else 0.0
            self_pct = s["self_sec"] / total_wall * 100
            label = f"{prefix}{connector}{name}"
            lines = [
                f"{label:<{col + len(prefix)}} {s['calls']:>6}  "
                f"{s['total_sec']:>10.3f}  {s['self_sec']:>10.3f}  {self_pct:>7.1f}%  "
                f"{s['mean_sec']:>10.3f}  {min_s:>10.3f}  {s['max_sec']:>10.3f}"
            ]
            children = edges.get(name, [])
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                lines += _row(child, child_prefix, i == len(children) - 1)
            return lines

        lines = []
        for i, root in enumerate(roots):
            s = stats[root]
            min_s = s["min_sec"] if s["min_sec"] is not None else 0.0
            self_pct = s["self_sec"] / total_wall * 100
            label = root
            lines.append(
                f"{label:<{col}} {s['calls']:>6}  "
                f"{s['total_sec']:>10.3f}  {s['self_sec']:>10.3f}  {self_pct:>7.1f}%  "
                f"{s['mean_sec']:>10.3f}  {min_s:>10.3f}  {s['max_sec']:>10.3f}"
            )
            children = edges.get(root, [])
            for j, child in enumerate(children):
                lines += _row(child, "", j == len(children) - 1)
            if i < len(roots) - 1:
                lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        with self._lock:
            funcs = {k: v.to_dict() for k, v in sorted(
                self._stats.items(), key=lambda x: -x[1].total_sec
            )}
            edges = {k: sorted(v) for k, v in self._edges.items()}
        return {"session": self.name, "functions": funcs, "call_tree": edges}

    def summary_text(self) -> str:
        col = 44
        header = (
            f"{'Function':<{col}} {'Calls':>6}  "
            f"{'Total(s)':>10}  {'Self(s)':>10}  {'Self%':>8}  "
            f"{'Mean(s)':>10}  {'Min(s)':>10}  {'Max(s)':>10}"
        )
        sep = "-" * len(header)
        lines = [
            f"Profile session: {self.name}",
            sep,
            header,
            sep,
        ]
        lines += self._tree_lines()
        lines.append(sep)
        return "\n".join(lines)

    def save(self, output_dir: str) -> None:
        """Write profile.json and profile.txt into output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "profile.txt"), "w", encoding="utf-8") as f:
            f.write(self.summary_text() + "\n")

    def __str__(self) -> str:
        return self.summary_text()

    def __repr__(self) -> str:
        n = len(self._stats)
        return f"ProfileSession(name={self.name!r}, functions={n}, enabled={self.enabled})"
