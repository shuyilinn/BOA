from __future__ import annotations

from collections import OrderedDict
import heapq
from typing import List, Optional, Tuple


class TrieNode:
    def __init__(self):
        self.children = {}
        # Store score only to save memory
        self.score = None
        # Cached subtree metadata for query pruning.
        self.max_score = None
        self.max_terminal_depth = -1


class SequenceCache:
    def __init__(self, max_entries: Optional[int] = None):
        self.root = TrieNode()
        self.max_entries = int(max_entries) if max_entries is not None else None
        # LRU of terminal sequences (key=tuple(token_ids)).
        self._lru: "OrderedDict[Tuple[int, ...], None]" = OrderedDict()

    def add(self, seq_ids: list, score: float):
        if not seq_ids:
            return
        seq_key = tuple(int(x) for x in seq_ids)
        node = self.root
        path_nodes = [self.root]
        for token in seq_key:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            path_nodes.append(node)
        # Mark path end with score
        node.score = score
        for depth in range(len(path_nodes) - 1, -1, -1):
            self._recompute_node_metadata(path_nodes[depth], depth)
        self._touch_lru(seq_key)
        self._evict_if_needed()

    def get(
        self,
        prefix_ids: list,
        *,
        top_k: Optional[int] = None,
        min_suffix_tokens: int = 0,
    ):
        if top_k is not None and int(top_k) <= 0:
            return []
        node = self.root

        # 1. Walk to prefix end
        for token in prefix_ids:
            if token not in node.children:
                return []
            node = node.children[token]

        results: List[Tuple[List[int], float]] = []
        prefix_len = len(prefix_ids)
        need_suffix = max(0, int(min_suffix_tokens or 0))
        min_total_len = prefix_len + need_suffix
        if node.max_terminal_depth < min_total_len:
            return []
        k_val = int(top_k) if top_k is not None else None

        # 2. DFS from prefix; pass copy of prefix_ids
        if k_val is None:
            self._dfs_all(node, list(prefix_ids), results, prefix_len, need_suffix, min_total_len)
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        heap: List[Tuple[float, List[int]]] = []
        self._dfs_top_k(node, list(prefix_ids), heap, k_val, prefix_len, need_suffix, min_total_len)
        return [(seq, score) for score, seq in sorted(heap, key=lambda x: x[0], reverse=True)]

    def _dfs_all(
        self,
        node: TrieNode,
        current_path: List[int],
        results: List[Tuple[List[int], float]],
        prefix_len: int,
        min_suffix_tokens: int,
        min_total_len: int,
    ) -> None:
        if node.max_terminal_depth < min_total_len:
            return
        if node.score is not None:
            suffix_len = len(current_path) - prefix_len
            if suffix_len >= min_suffix_tokens:
                # Copy current_path (it is mutated during DFS)
                results.append((list(current_path), node.score))

        for token, child_node in node.children.items():
            current_path.append(token)
            self._dfs_all(
                child_node,
                current_path,
                results,
                prefix_len,
                min_suffix_tokens,
                min_total_len,
            )
            current_path.pop()

    def _dfs_top_k(
        self,
        node: TrieNode,
        current_path: List[int],
        heap: List[Tuple[float, List[int]]],
        k_val: int,
        prefix_len: int,
        min_suffix_tokens: int,
        min_total_len: int,
    ) -> None:
        if node.max_terminal_depth < min_total_len:
            return
        if len(heap) >= k_val:
            node_upper = float(node.max_score) if node.max_score is not None else float("-inf")
            if node_upper <= heap[0][0]:
                return
        if node.score is not None:
            suffix_len = len(current_path) - prefix_len
            if suffix_len >= min_suffix_tokens:
                entry = (float(node.score), list(current_path))
                if len(heap) < k_val:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)

        children = list(node.children.items())
        children.sort(
            key=lambda kv: (float(kv[1].max_score) if kv[1].max_score is not None else float("-inf")),
            reverse=True,
        )
        for token, child_node in children:
            current_path.append(token)
            self._dfs_top_k(
                child_node,
                current_path,
                heap,
                k_val,
                prefix_len,
                min_suffix_tokens,
                min_total_len,
            )
            current_path.pop()

    def _touch_lru(self, seq_key: Tuple[int, ...]) -> None:
        if seq_key in self._lru:
            self._lru.move_to_end(seq_key)
        else:
            self._lru[seq_key] = None

    def _evict_if_needed(self) -> None:
        if self.max_entries is None or self.max_entries <= 0:
            return
        while len(self._lru) > self.max_entries:
            old_seq, _ = self._lru.popitem(last=False)
            self._remove_sequence(old_seq)

    def _remove_sequence(self, seq_key: Tuple[int, ...]) -> None:
        # Walk path while keeping parents for upward pruning.
        node = self.root
        path_nodes: List[TrieNode] = [self.root]
        stack: List[Tuple[TrieNode, int]] = []
        for token in seq_key:
            child = node.children.get(token)
            if child is None:
                return
            stack.append((node, token))
            node = child
            path_nodes.append(node)

        if node.score is None:
            return
        node.score = None

        # Remove empty nodes bottom-up.
        for parent, token in reversed(stack):
            child = parent.children.get(token)
            if child is None:
                continue
            if child.score is None and not child.children:
                del parent.children[token]
            else:
                break

        # Refresh metadata for impacted prefix nodes.
        for depth in range(len(path_nodes) - 1, -1, -1):
            self._recompute_node_metadata(path_nodes[depth], depth)

    def _recompute_node_metadata(self, node: TrieNode, node_depth: int) -> None:
        best_score = float(node.score) if node.score is not None else None
        deepest_terminal = node_depth if node.score is not None else -1
        for child in node.children.values():
            child_score = child.max_score
            if child_score is not None and (best_score is None or float(child_score) > best_score):
                best_score = float(child_score)
            if child.max_terminal_depth > deepest_terminal:
                deepest_terminal = int(child.max_terminal_depth)
        node.max_score = best_score
        node.max_terminal_depth = deepest_terminal
