from __future__ import annotations

import time
from typing import Optional


class RuntimeGuard:
    """
    Runtime stop-condition guard (time/token/depth/node budgets).
    """

    def __init__(
        self,
        *,
        time_limit_sec: Optional[float],
        token_limit: Optional[int],
        depth_limit: Optional[int],
        node_limit: Optional[int],
        logger,
    ):
        self.time_limit_sec = time_limit_sec
        self.token_limit = token_limit
        self.depth_limit = depth_limit
        self.node_limit = node_limit
        self.logger = logger
        self.timeout_reached = False
        self.timeout_reason: Optional[str] = None
        self.token_budget_reached = False
        self.token_budget_reason: Optional[str] = None
        self.depth_budget_reached = False
        self.depth_budget_reason: Optional[str] = None
        self.node_budget_reached = False
        self.node_budget_reason: Optional[str] = None

    def check_timeout(self, checkpoint: str, start_time: float) -> bool:
        if self.timeout_reached:
            return True

        limit = self.time_limit_sec
        if limit is None:
            return False
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            return False

        elapsed = time.time() - float(start_time)
        if elapsed < limit:
            return False

        self.timeout_reached = True
        self.timeout_reason = (
            f"time_limit_sec exceeded at {checkpoint}: elapsed={elapsed:.2f}s, limit={limit:.2f}s"
        )
        self.logger.info("Stopping search due to timeout: %s", self.timeout_reason)
        return True

    def check_token_budget(self, checkpoint: str, used_tokens: int) -> bool:
        if self.token_budget_reached:
            return True

        limit = self.token_limit
        if limit is None:
            return False
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return False
        if limit <= 0:
            return False

        used = int(used_tokens or 0)
        if used < limit:
            return False

        self.token_budget_reached = True
        self.token_budget_reason = (
            f"token_limit exceeded at {checkpoint}: used={used}, limit={limit}"
        )
        self.logger.info("Stopping search due to token budget: %s", self.token_budget_reason)
        return True

    def check_depth_budget(self, checkpoint: str, max_evaluated_depth: int) -> bool:
        if self.depth_budget_reached:
            return True

        limit = self.depth_limit
        if limit is None:
            return False
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return False
        if limit < 0:
            return False

        used = int(max_evaluated_depth)
        if used < limit:
            return False

        self.depth_budget_reached = True
        self.depth_budget_reason = (
            f"depth_limit reached at {checkpoint}: max_evaluated_depth={used}, limit={limit}"
        )
        self.logger.info("Stopping search due to depth budget: %s", self.depth_budget_reason)
        return True

    def check_node_budget(self, checkpoint: str, evaluated_nodes_count: int) -> bool:
        if self.node_budget_reached:
            return True

        limit = self.node_limit
        if limit is None:
            return False
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return False
        if limit <= 0:
            return False

        used = int(evaluated_nodes_count or 0)
        if used < limit:
            return False

        self.node_budget_reached = True
        self.node_budget_reason = (
            f"node_limit reached at {checkpoint}: evaluated_nodes_count={used}, limit={limit}"
        )
        self.logger.info("Stopping search due to node budget: %s", self.node_budget_reason)
        return True

    def any_budget_reached(self) -> bool:
        return (
            self.timeout_reached
            or self.token_budget_reached
            or self.depth_budget_reached
            or self.node_budget_reached
        )

    def check_runtime_limits(
        self,
        checkpoint: str,
        *,
        start_time: float,
        used_tokens: int,
        max_evaluated_depth: int,
        evaluated_nodes_count: int,
    ) -> bool:
        return (
            self.check_timeout(checkpoint, start_time)
            or self.check_token_budget(checkpoint, used_tokens)
            or self.check_depth_budget(checkpoint, max_evaluated_depth)
            or self.check_node_budget(checkpoint, evaluated_nodes_count)
        )
