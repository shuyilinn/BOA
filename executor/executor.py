from config import Config
from utils.logger import setup_logger

from boa_types.tree_node import TreeNode, NodeStatus, NodeSource
from searchers.naive_searcher import NaiveSearcher
from sampler.l2_expander import L2Expander
from sampler.sampler import Sampler
from components.buffer.buffer import Buffer
from judgers.factory import create_judger
from components.cache.cache import SequenceCache as Cache
from reporters.reporter import Reporter
from executor.engine_factory import create_api_judger_engine, create_judger_engine, create_target_engine
from executor.sample_worker import SampleWorker
from executor.judge_worker import JudgeWorker
from executor.routing_policy import RoutingPolicy
from executor.metrics import buffer_stats, cache_stats
from executor.debug_view import node_brief, build_tree_snapshot
from executor.runtime_guard import RuntimeGuard
from executor.result_builder import build_final_stats
from utils.batch_policy import RuntimeOOMBatchRunner
from profiler import ProfileSession, set_session, clear_session, set_torch_profile_dir
import time
import os
import json
from typing import Optional, List, Dict, Any

logger = setup_logger("Executor")


class Executor:
    def __init__(
        self,
        config: Config,
        threshold: Optional[List[float]] = None,
        init_judger: bool = True,
        init_target: bool = True,
    ):
        self.config = config
        self.threshold = threshold
        self._enable_sampling_cache = self.config.enable_sampling_cache

        self.target_model_engine = None
        self.expander = None
        self.sampler = None
        self.sample_worker = None

        self.judger_engine = None
        self.api_judger_engine = None
        self.local_judger_engine = None
        self.judger = None
        self.judge_worker = None

        self._target_initialized = False
        self._judger_initialized = False

        self.routing_policy = RoutingPolicy(self.config)
        self.reporter = Reporter(self.config)
        self._sampler_batch_size: Optional[int] = None
        self._judger_batch_size: Optional[int] = None
        self._attack_sample_oom_runner: Optional[RuntimeOOMBatchRunner] = None
        self._attack_judge_oom_runner: Optional[RuntimeOOMBatchRunner] = None

        if init_target:
            self.initialize_target_components()
        else:
            logger.info("Target initialization deferred.")

        if init_judger:
            self.initialize_judger_components()
        else:
            logger.info("Judger initialization deferred until after threshold baseline.")

    def initialize_target_components(self) -> None:
        if self._target_initialized:
            logger.info("Target components already initialized; skip.")
            return

        logger.info("Init target components...")
        self.target_model_engine = create_target_engine(self.config)
        self.expander = L2Expander(
            self.target_model_engine,
            self.config,
            threshold=self.threshold,
        )
        self.sampler = Sampler(self.target_model_engine, self.config)
        self.sampler.log_probability_threshold = self.threshold
        self.sample_worker = SampleWorker(self.sampler)

        self._target_initialized = True
        logger.info(
            "Target engine setup complete: target_engine=%s test_mode=%s",
            type(self.target_model_engine).__name__,
            bool(getattr(self.config, "test_mode", False)),
        )

    def initialize_judger_components(self) -> None:
        if self._judger_initialized:
            logger.info("Judger components already initialized; skip.")
            return
        if not self._target_initialized:
            self.initialize_target_components()

        logger.info("Init judger components...")
        self.judger_engine = create_judger_engine(self.config)
        self.api_judger_engine = create_api_judger_engine(self.config)
        self.local_judger_engine = self.judger_engine
        self.judger = create_judger(
            config=self.config,
            local_judger_engine=self.local_judger_engine,
            api_judger_engine=self.api_judger_engine,
        )
        self.judge_worker = JudgeWorker(
            self.judger,
            self.target_model_engine,
            self.sampler,
            self.config,
        )

        self._judger_initialized = True
        logger.info(
            "Judger setup complete: judger_engine=%s",
            type(self.judger_engine).__name__,
        )

    def set_threshold(self, threshold: Optional[List[float]]) -> None:
        self.threshold = threshold
        if self.expander is not None and getattr(self.expander, "l3_handler", None) is not None:
            self.expander.l3_handler.threshold = threshold
        if self.sampler is not None:
            self.sampler.log_probability_threshold = threshold
        logger.info("Set threshold: %s", "None" if threshold is None else "loaded/built")

    def run(
        self,
        prompt: str,
        original_prompt: str,
        prompt_metadata: Optional[Dict[str, Any]] = None,
        prompt_index: Optional[int] = None,
        total_prompts: Optional[int] = None,
    ):
        self.initialize_judger_components()
        self._step_idx = 0
        run_id = getattr(self.reporter, "run_id", "run")
        self.profile_session = ProfileSession(name=run_id, enabled=bool(getattr(self.config, "enable_profiling", True)))
        set_session(self.profile_session)
        if int(getattr(self.config, "torch_profiler_steps", 0)) > 0:
            set_torch_profile_dir(self.reporter.result_dir)
        self._cache_lookups = 0
        self._cache_partial_hits = 0
        self._cache_full_hits = 0
        self._sample_flush_batch_sizes = []
        self._judging_flush_batch_sizes = []
        self._sample_enqueued_items = 0
        self._judging_enqueued_items = 0
        self._sample_buffer_max_size = 0
        self._judging_buffer_max_size = 0
        self._diag = {
            "process_buffer_calls": 0,
            "route_judge_calls": 0,
            "route_judge_tasks_total": 0,
            "expand_ms_total": 0.0,
            "add_requests_ms_total": 0.0,
            "drain_ms_total": 0.0,
            "route_judge_ms_total": 0.0,
        }
        self._source_token_counts = {
            "attack_sampling": 0,
            "sampling": 0,
            "full_response_extend": 0,
        }
        self.judger_prompt = prompt
        self.original_prompt = original_prompt
        self.prompt_metadata = dict(prompt_metadata or {})
        self.stats = {
            "start_time": time.time(),
            "prompt": prompt,
            "original_prompt": original_prompt,
            "prompt_metadata": dict(self.prompt_metadata),
            "prompt_index": prompt_index,
            "total_prompts": total_prompts,
            "total_tokens_generated": 0,
            "token_sources": dict(self._source_token_counts),
            "evaluated_nodes_count": 0,
            "max_evaluated_depth": -1,
            "log_probability": None,
            "probability": None,
        }
        self._total_tokens_generated = 0
        self._evaluated_nodes_count = 0
        self._max_evaluated_depth = -1
        self.runtime_guard = RuntimeGuard(
            time_limit_sec=self.config.time_limit_sec,
            token_limit=self.config.token_limit,
            depth_limit=self.config.depth_limit,
            node_limit=self.config.node_limit,
            logger=logger,
        )
        self.searcher = NaiveSearcher(max_beam_width=20000)
        # initialize bounded cache to avoid unbounded growth on deep trees
        self.cache = Cache(max_entries=self.config.cache_max_entries)
        
        # initialize the buffer
        # apply chat template if needed
        prompt_with_chat_template, prompt_with_chat_template_ids = self.apply_chat_template(prompt)
        self.prompt_with_chat_template = prompt_with_chat_template
        self.stats["effective_prompt"] = prompt_with_chat_template
        sampler_bs = self.sampler.get_batch_size()
        judger_bs = self.judger.get_batch_size()
        self._sampler_batch_size = int(sampler_bs)
        self._judger_batch_size = int(judger_bs)
        self.sample_buffer = Buffer(capacity=self.config.buffer_capacity)
        self.judging_buffer = Buffer(capacity=self.config.buffer_capacity)
        use_dynamic = self.config.use_dynamic_batch_size
        logger.info(
            "Batch size: use_dynamic=%s, sampler=%s, judger=%s",
            use_dynamic,
            self._sampler_batch_size,
            self._judger_batch_size,
        )
        logger.info(
            "Buffers initialized: capacity=%s, flush_threshold: sample=%s judging=%s",
            self.config.buffer_capacity,
            self._sampler_batch_size,
            self._judger_batch_size,
        )
        logger.info("Starting execution for prompt: %s", prompt)
        logger.info("=" * 72)
        logger.info("PROMPT: %s", prompt)
        logger.info("=" * 72)

        
 
        # here, the root should be nothing, and when we calculate the path, we ignore the root.
        root = TreeNode(
            token_ids=prompt_with_chat_template_ids,
            text=prompt_with_chat_template,
            log_prob=0.0,
            cum_log_prob=0.0,
            depth=0,
            status=NodeStatus.CREATED,
            source=NodeSource.ROOT,
            environment_type=self.config.workload_configs[self.config.workload_name]["environment_type"],
            prompt_metadata=dict(self.prompt_metadata),
        )
        self.root = root
        self._nodes_created = 1
        self._max_depth_seen = root.depth
        self.searcher.add_node(root)
        logger.info("Root queued: %s", self._node_brief(root))
        # --- Phase 1: Attack sampling (optional) ---
        if not self.config.enable_attack_sampling:
            logger.info("Attack sampling disabled (enable_attack_sampling=False).")
            sampler_result = None
        else:
            sampler_result = self.attack_sampling(root)
        logger.info("Sampler result: %s", sampler_result)
        if sampler_result is not None:
            self._finalize_success(sampler_result)
            return sampler_result

        # --- Phase 2: Tree search loop ---
        while True:
            # Guard before starting any new step.
            if self._check_runtime_limits("loop_start"):
                break
            # 1. Pick a node
            node = self.searcher.select_next_node()

            # if there is no more node, then flush the buffer
            if node is None:
                logger.info(
                    "Frontier empty (no expandable node in searcher priority_queue). "
                    "Pending buffer tasks: sample_buffer=%s, judging_buffer=%s. "
                    "Executor will flush buffers before next selection.",
                    len(self.sample_buffer),
                    len(self.judging_buffer),
                )
                if not self.sample_buffer.is_empty() or not self.judging_buffer.is_empty():
                    success_node = self.process_buffer()
                    if success_node:
                        self._finalize_success(success_node)
                        return success_node
                    if self.runtime_guard.any_budget_reached():
                        break
                    continue
                else:
                    break
            self._step_idx += 1
            self._log_step_banner()
            logger.info("Selected node: %s", self._node_brief(node))


            # TODO: optimize: collect many nodes and expand the node in parallel
            # expand the node and generate its children.
            # Expander only produces children structure/content; status/score/buffer routing by Executor/Judger. See CONVENTIONS.md.
            node.status = NodeStatus.EXPANDING
            t_expand = time.monotonic()
            new_nodes = self.expander.expand(node)
            self._diag["expand_ms_total"] += (time.monotonic() - t_expand) * 1000.0
            self._nodes_created += len(new_nodes)
            if new_nodes:
                self._max_depth_seen = max(
                    self._max_depth_seen,
                    max(getattr(child, "depth", 0) for child in new_nodes),
                )
            if self._check_runtime_limits("post_expand"):
                break
            logger.info(
                "Expanded node depth=%s into %s children",
                node.depth,
                len(new_nodes),
            )
            for child in new_nodes:
                # Keep tau-pruned nodes on tree, but mark as CUT and skip routing.
                if bool(getattr(child, "metadata", {}).get("is_cut", False)):
                    child.status = NodeStatus.CUT
                    child.is_cut = True
                elif child.metadata.get("should_complete"):
                    self._mark_completed_and_bubble(child)
                else:
                    child.status = NodeStatus.CREATED
                    child.scores = []
            self._mark_node_evaluated(node)


            target_count = self.config.sampler_number
            for child in new_nodes:
                if child.status in (NodeStatus.CUT, NodeStatus.COMPLETED, NodeStatus.JAILBREAKED):
                    continue
                # Process the cache
                cached_children = []
                if self._enable_sampling_cache:
                    path_ids = child.get_path_token_ids()
                    min_suffix_tokens = self.config.cache_min_suffix_tokens
                    cached_children = self.cache.get(
                        path_ids,
                        top_k=target_count,
                        min_suffix_tokens=min_suffix_tokens,
                    ) or []
                    self._cache_lookups += 1
                    for cached in cached_children:
                        child.scores.append(cached[1])

                # Compute how many sequences are still needed
                needed_count = max(0, target_count - len(cached_children))
                if len(cached_children) > 0:
                    if needed_count == 0:
                        self._cache_full_hits += 1
                    else:
                        self._cache_partial_hits += 1

                # 4. Routing logic
                if needed_count > 0:
                    # A. Still need generation: push into Sample Buffer
                    # Pack "parent node" + "missing count" as a task
                    t_add = time.monotonic()
                    self.sample_buffer.add_requests(
                        child,
                        needed_count,
                        judger_prompt=self.judger_prompt,
                        judger_metadata=self.prompt_metadata,
                        original_prompt=self.original_prompt,
                        prompt_with_chat_template=self.prompt_with_chat_template,
                    )
                    self._diag["add_requests_ms_total"] += (time.monotonic() - t_add) * 1000.0
                    self._sample_enqueued_items += needed_count
                    self._sample_buffer_max_size = max(self._sample_buffer_max_size, len(self.sample_buffer))
                    logger.debug(
                        "Queued sample task in sample_buffer: %s missing_samples=%s cached_samples=%s sample_buffer_size=%s",
                        self._node_brief(child),
                        needed_count,
                        len(cached_children),
                        len(self.sample_buffer),
                    )
        
                    if len(self.sample_buffer) >= self._sampler_batch_size:
                        judge_result = self.process_buffer()
                        if judge_result is not None:
                            self._finalize_success(judge_result)
                            return judge_result
                        if self.runtime_guard.any_budget_reached():
                            break
                else:
                    # B. Cache is sufficient: skip Sampler and push into Searcher directly
                    # (Note: if cached nodes are not scored yet, they should go to judging_buffer;
                    #  here we assume cache stores finished nodes.)
                    child.score = float(sum(child.scores) / len(child.scores)) if child.scores else 0.0
                    self.searcher.add_node(child)
                    logger.info(
                        "Cache full hit -> queued child: %s cached=%s/%s",
                        self._node_brief(child),
                        len(cached_children),
                        target_count,
                    )

            # 5. Opportunistic flush of Judging Buffer
            # Avoid the case where Sample Buffer is not full but Judging Buffer has built up a lot.
            if len(self.judging_buffer) >= self._judger_batch_size:
                judge_result = self.process_judging_only()
                if judge_result is not None:
                    self._finalize_success(judge_result)
                    return judge_result
                if self.runtime_guard.any_budget_reached():
                    break
            if self._step_idx % 10 == 0:
                self._log_tree_snapshot()
        # Here means no unsafe
        self.success_callback(None, None, exit_reason=None)

    def process_buffer(self):
        """
        Cascading processing: Sample Buffer -> Judging Buffer -> Searcher
        """
        self._diag["process_buffer_calls"] += 1
        t0 = time.monotonic()

        # --- Stage 1: Sample -> Judge ---
        if not self.sample_buffer.is_empty():
            self._consume_sample_once()


        # --- Stage 2: Judge -> Searcher ---
        # As long as Judging Buffer has items and (forced flush or full), process it.
        # Note: Stage 1 may have just pushed items here, so this will likely trigger.
        if not self.judging_buffer.is_empty():
            judge_result = self.process_judging_only()
            if judge_result is not None:
                self._diag["drain_ms_total"] += (time.monotonic() - t0) * 1000.0
                return judge_result

        self._diag["drain_ms_total"] += (time.monotonic() - t0) * 1000.0
        return None

    def _consume_sample_once(self) -> bool:
        result = self.sample_worker.flush_once(
            sample_buffer=self.sample_buffer,
            judging_buffer=self.judging_buffer,
        )
        return self._apply_sample_batch_result(result)

    def _apply_sample_batch_result(self, result) -> bool:
        if result.tasks <= 0:
            return False
        self._sample_flush_batch_sizes.append(int(result.tasks))
        self._add_generated_tokens(int(result.generated_tokens), source="sampling")
        self._judging_enqueued_items += int(result.pushed_to_judging)
        self._sample_enqueued_items += int(result.requeued_to_sample)
        self._sample_buffer_max_size = max(self._sample_buffer_max_size, len(self.sample_buffer))
        self._judging_buffer_max_size = max(self._judging_buffer_max_size, len(self.judging_buffer))
        return True

    def process_judging_only(self):
        """Flush and process Judging Buffer."""
        while not self.judging_buffer.is_empty():
            maybe_success = self._consume_judge_once()
            if maybe_success is not None:
                return maybe_success
        return None

    def _consume_judge_once(self):
        judge_result = self.judge_worker.flush_once(
            judging_buffer=self.judging_buffer,
            batch_size=self._judger_batch_size,
            node_brief_fn=self._node_brief,
        )
        return self._route_judge_batch_result(judge_result)

    def _route_judge_batch_result(self, judge_result):
        self._diag["route_judge_calls"] += 1
        self._diag["route_judge_tasks_total"] += len(judge_result.tasks)
        t0 = time.monotonic()
        out = self.routing_policy.apply_judge_batch_result(
            judge_result,
            cache=self.cache,
            node_brief_fn=self._node_brief,
            check_runtime_limits_fn=self._check_runtime_limits,
            add_generated_tokens_fn=self._add_generated_tokens,
            make_success_payload_fn=self._make_success_payload,
            on_judging_flush=lambda n: self._judging_flush_batch_sizes.append(int(n)),
            finalize_rollout_node_fn=self._finalize_rollout_node_if_ready,
        )
        self._diag["route_judge_ms_total"] += (time.monotonic() - t0) * 1000.0
        return out

    def _finalize_rollout_node_if_ready(self, node: TreeNode, source: str) -> bool:
        """
        Finalize node rollout when enough judged samples are collected.
        Returns True if node was finalized (queued or completed), else False.
        """
        if node.status in (
            NodeStatus.QUEUED,
            NodeStatus.EXPLORING,
            NodeStatus.COMPLETED,
            NodeStatus.CUT,
            NodeStatus.JAILBREAKED,
        ):
            return False

        target_count = self.config.sampler_number
        scores = node.scores
        if len(scores) < target_count:
            return False

        node.score = float(sum(scores) / len(scores)) if scores else 0.0

        self.searcher.add_node(node)
        logger.info(
            "Rollout finalized -> queued node: %s scores=%s/%s agg=%.2f",
            self._node_brief(node),
            len(scores),
            target_count,
            float(node.score),
        )
        return True

    def _node_brief(self, node: TreeNode) -> str:
        return node_brief(node)

    def _log_step_banner(self) -> None:
        cache_stats_payload = cache_stats(
            lookups=self._cache_lookups,
            partial_hits=self._cache_partial_hits,
            full_hits=self._cache_full_hits,
        )
        logger.info(
            "-" * 72 + " STEP %s | frontier=%s sample_buf=%s judge_buf=%s | cache full=%.1f%% any=%.1f%% (%s/%s)",
            self._step_idx,
            len(self.searcher.priority_queue),
            len(self.sample_buffer),
            len(self.judging_buffer),
            cache_stats_payload["full_hit_rate_pct"],
            cache_stats_payload["any_hit_rate_pct"],
            cache_stats_payload["full_hits"],
            cache_stats_payload["lookups"],
        )

    def _log_tree_snapshot(self) -> None:
        """
        Lightweight tree snapshot for runtime visibility.
        Limits depth and node count to avoid flooding logs.
        """
        tree_stats = {
            "total": self._nodes_created,
            "max_depth": self._max_depth_seen,
            "evaluated": self._evaluated_nodes_count,
            "queued": len(self.searcher.priority_queue),
        }
        self.stats["tree_stats"] = tree_stats
        logger.info(
            "Tree stats: total=%s max_depth=%s evaluated=%s queued=%s",
            tree_stats["total"],
            tree_stats["max_depth"],
            tree_stats["evaluated"],
            tree_stats["queued"],
        )
        max_depth = self.config.logger_tree_max_depth
        max_nodes = self.config.logger_tree_max_nodes
        snapshot = build_tree_snapshot(self.root, max_depth=max_depth, max_nodes=max_nodes)
        if not snapshot:
            return
        logger.info("Tree snapshot (depth<=%s, nodes<=%s):\n%s", max_depth, max_nodes, snapshot)

    def _mark_node_evaluated(self, node: TreeNode) -> None:
        if node.status != NodeStatus.EVALUATED:
            self._evaluated_nodes_count += 1
            self._max_evaluated_depth = max(self._max_evaluated_depth, node.depth)
        node.status = NodeStatus.EVALUATED

    def _mark_completed_and_bubble(self, node: TreeNode) -> None:
        if node.status != NodeStatus.JAILBREAKED:
            node.status = NodeStatus.COMPLETED
        parent = node.parent
        while parent is not None:
            children = parent.children
            if not children:
                break
            if all(c.status == NodeStatus.COMPLETED for c in children):
                if parent.status == NodeStatus.JAILBREAKED:
                    break
                parent.status = NodeStatus.COMPLETED
                parent = parent.parent
                continue
            break

    def _check_token_budget(self, checkpoint: str) -> bool:
        return self.runtime_guard.check_token_budget(checkpoint, self._total_tokens_generated)

    def _check_runtime_limits(self, checkpoint: str) -> bool:
        return self.runtime_guard.check_runtime_limits(
            checkpoint,
            start_time=float(self.stats.get("start_time", time.time())),
            used_tokens=self._total_tokens_generated,
            max_evaluated_depth=int(getattr(self, "_max_evaluated_depth", -1)),
            evaluated_nodes_count=int(getattr(self, "_evaluated_nodes_count", 0)),
        )



    def attack_sampling(self, root):
        attack_sampler_number = self.config.attack_sampler_number
        sequences = [root.get_path_token_ids() for _ in range(attack_sampler_number)]
        attack_max_new_tokens = self.config.attack_sample_new_tokens
        logger.info(
            "Attack sampling start: samples=%s, max_new_tokens=%s, threshold_enabled=%s",
            attack_sampler_number,
            attack_max_new_tokens,
            bool(self.threshold),
        )
        sampler_bs = self._sampler_batch_size
        judger_bs = self._judger_batch_size
        if self._attack_sample_oom_runner is None:
            self._attack_sample_oom_runner = RuntimeOOMBatchRunner(
                initial_batch_size=max(1, sampler_bs),
                logger=logger,
                policy_name="AttackSampling-Sampler",
            )
        if self._attack_judge_oom_runner is None:
            self._attack_judge_oom_runner = RuntimeOOMBatchRunner(
                initial_batch_size=max(1, judger_bs),
                logger=logger,
                policy_name="AttackSampling-Judger",
            )
        # Tau constraint is enforced inside sampler/customized_sampler.
        # Do not apply a second tau filter at this level.
        new_children_groups: List[List[int]] = []

        def _sample_chunk(chunk_sequences: List[List[int]]) -> None:
            chunk_out = self.sampler.batch_uniform_generate(
                chunk_sequences,
                max_new_tokens=attack_max_new_tokens,
            )
            new_children_groups.extend(chunk_out)

        self._attack_sample_oom_runner.run(sequences, _sample_chunk)
        self._add_generated_tokens(
            sum(len(ids) for ids in new_children_groups),
            source="attack_sampling",
        )
        logger.info("Attack sampling generated candidates=%s", len(new_children_groups))

        if not new_children_groups:
            logger.info("Attack sampling produced no usable candidates.")
            return None
        prompt_text = self.judger_prompt or self.original_prompt or root.get_path_text()
        response_texts = []
        for ids in new_children_groups:
            full_text = self.sampler.tokenizer.decode(ids)
            if self.prompt_with_chat_template and full_text.startswith(self.prompt_with_chat_template):
                response_texts.append(full_text[len(self.prompt_with_chat_template):])
            else:
                response_texts.append(full_text)
        judger_results: List[Dict[str, Any]] = []

        pairs = list(zip([prompt_text] * len(response_texts), response_texts))

        def _judge_chunk(chunk_pairs: List[Any]) -> List[Dict[str, Any]]:
            chunk_prompts = [p for p, _ in chunk_pairs]
            chunk_responses = [r for _, r in chunk_pairs]
            chunk_metadatas = [dict(self.prompt_metadata) for _ in chunk_pairs]
            return self.judger.batch_evaluate_attack_sampling(
                prompts=chunk_prompts,
                responses=chunk_responses,
                metadatas=chunk_metadatas,
            )

        for chunk_res in self._attack_judge_oom_runner.run(pairs, _judge_chunk):
            judger_results.extend(chunk_res)
        layer_hist = {}
        unsafe_count = 0
        best_score = None
        for jr in judger_results:
            layer = int(jr.get("layer", -1))
            layer_hist[layer] = layer_hist.get(layer, 0) + 1
            score = jr.get("score")
            if isinstance(score, (int, float)):
                best_score = float(score) if best_score is None else max(best_score, float(score))
            if not bool(jr.get("is_safe", True)):
                unsafe_count += 1
        logger.info(
            "Attack sampling judged: candidates=%s, layer_hist=%s, unsafe=%s, best_score=%s",
            len(judger_results),
            layer_hist,
            unsafe_count,
            "N/A" if best_score is None else f"{best_score:.2f}",
        )
        for judger_result in judger_results:
            if self.routing_policy.is_terminal_unsafe(judger_result):
                hit_score = float(judger_result.get("score", 0.0))
                # Attack-sampling success returns root directly; persist score for reporting.
                root.score = hit_score
                if hasattr(root, "scores") and isinstance(getattr(root, "scores"), list):
                    root.scores.append(hit_score)
                logger.warning(
                    "!!! TERMINAL UNSAFE HIT (attack_sampling): root=%s layer=%s score=%.2f !!!",
                    self._node_brief(root),
                    int(judger_result.get("layer", -1)),
                    hit_score,
                )
                return self.routing_policy.make_terminal_success_payload(
                    node=root,
                    result=judger_result,
                    make_success_payload_fn=self._make_success_payload,
                    exit_reason="unsafe_detected_attack_sampling",
                )
        logger.info("Attack sampling finished: no terminal unsafe hit, fallback to tree search.")
        return None





    def _make_success_payload(
        self,
        node: Optional[TreeNode],
        unsafe_result: Optional[str],
        response_score: Optional[float] = None,
        layer3_score: Optional[float] = None,
        layer4_score: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "node": node,
            "unsafe_result": unsafe_result,
            "response_score": response_score,
            "layer3_score": layer3_score,
            "layer4_score": layer4_score,
            "exit_reason": exit_reason,
        }

    def _add_generated_tokens(self, n: int, source: str) -> None:
        self._total_tokens_generated += n
        self.stats["total_tokens_generated"] = self._total_tokens_generated
        self._source_token_counts[source] = self._source_token_counts.get(source, 0) + n
        self.stats["token_sources"] = dict(self._source_token_counts)
        self._check_token_budget(f"token_update:{source}")
        logger.debug(
            "Token usage updated: +%s from %s (total=%s)",
            n,
            source,
            self._total_tokens_generated,
        )

    def _finalize_success(self, success: Dict[str, Any]) -> None:
        self._commit_success_payload(success)

    def _commit_success_payload(self, success: Dict[str, Any]) -> None:
        self.success_callback(
            success.get("node"),
            success.get("unsafe_result"),
            response_score=success.get("response_score"),
            layer3_score=success.get("layer3_score"),
            layer4_score=success.get("layer4_score"),
            exit_reason=success.get("exit_reason"),
        )

    def success_callback(
        self,
        node: Optional[TreeNode],
        unsafe_result: Optional[str],
        response_score: Optional[float] = None,
        layer3_score: Optional[float] = None,
        layer4_score: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ):
        # 1. Update stats
        logger.info("Success callback: node=%s unsafe_result=%s", node, unsafe_result)
        self.stats["evaluated_nodes_count"] = self._evaluated_nodes_count
        self.stats["max_evaluated_depth"] = self._max_evaluated_depth
        sample_flushed = sum(self._sample_flush_batch_sizes)
        judge_flushed = sum(self._judging_flush_batch_sizes)
        sample_residual = len(self.sample_buffer)
        judge_residual = len(self.judging_buffer)
        sample_conserved = self._sample_enqueued_items == (sample_flushed + sample_residual)
        judge_conserved = self._judging_enqueued_items == (judge_flushed + judge_residual)
        diagnostics = {
            "process_buffer_calls": int(self._diag["process_buffer_calls"]),
            "sample_flush_count": len(self._sample_flush_batch_sizes),
            "sample_tasks_processed": sample_flushed,
            "judge_flush_count": len(self._judging_flush_batch_sizes),
            "judge_tasks_processed": judge_flushed,
            "route_judge_calls": int(self._diag["route_judge_calls"]),
            "route_judge_tasks_total": int(self._diag["route_judge_tasks_total"]),
            "sample_buffer_conserved": bool(sample_conserved),
            "judge_buffer_conserved": bool(judge_conserved),
            "time_breakdown_ms": {
                "expand": float(self._diag["expand_ms_total"]),
                "add_requests": float(self._diag["add_requests_ms_total"]),
                "drain": float(self._diag["drain_ms_total"]),
                "route_judge": float(self._diag["route_judge_ms_total"]),
            },
        }
        self.stats["diagnostics"] = diagnostics
        refusal_filter_stats = {}
        if self.judger is not None and hasattr(self.judger, "get_refusal_filter_stats"):
            refusal_filter_stats = self.judger.get_refusal_filter_stats() or {}
        self.stats["refusal_filter"] = refusal_filter_stats
        cache_payload = cache_stats(
            lookups=self._cache_lookups,
            partial_hits=self._cache_partial_hits,
            full_hits=self._cache_full_hits,
        )
        buffer_payload = buffer_stats(
            sample_buffer_capacity=self.sample_buffer.capacity,
            sample_enqueued_items=self._sample_enqueued_items,
            sample_max_queue_size=self._sample_buffer_max_size,
            sample_flush_batch_sizes=self._sample_flush_batch_sizes,
            judging_buffer_capacity=self.judging_buffer.capacity,
            judging_enqueued_items=self._judging_enqueued_items,
            judging_max_queue_size=self._judging_buffer_max_size,
            judging_flush_batch_sizes=self._judging_flush_batch_sizes,
        )
        finalize_result = build_final_stats(
            stats=self.stats,
            node=node,
            unsafe_result=unsafe_result,
            response_score=response_score,
            layer3_score=layer3_score,
            layer4_score=layer4_score,
            exit_reason=exit_reason,
            runtime_guard=self.runtime_guard,
            duration_sec=(time.time() - self.stats["start_time"]),
            cache_payload=cache_payload,
            buffer_payload=buffer_payload,
        )
        if finalize_result.missing_unsafe_result:
            try:
                raise ValueError("success node exists but unsafe_result is empty")
            except ValueError:
                logger.exception("Invariant violation in success_callback")

        if finalize_result.jailbreak_found:
            logger.warning(
                "!!! SUCCESS NODE FOUND: %s final_output='%s' !!!",
                self._node_brief(node),
                finalize_result.final_output_text,
            )

        self.stats["tree_stats"] = {
            "total": self._nodes_created,
            "max_depth": self._max_depth_seen,
            "evaluated": self._evaluated_nodes_count,
            "queued": len(self.searcher.priority_queue),
        }
        logger.info(
            "Tree stats (persist): total=%s max_depth=%s evaluated=%s queued=%s",
            self.stats["tree_stats"]["total"],
            self.stats["tree_stats"]["max_depth"],
            self.stats["tree_stats"]["evaluated"],
            self.stats["tree_stats"]["queued"],
        )

        # 2. Save profiling data alongside other results
        if hasattr(self, "profile_session"):
            self.stats["profiling"] = self.profile_session.to_dict()
            self.profile_session.save(self.reporter.result_dir)
            clear_session()

        # 3. Reporter writes files under reporter's result_dir with a unique run id
        self.reporter.generate_reports(
            stats=self.stats,
            root_node=self.root,
        )
        cache_stats_data = self.stats.get("cache", {})
        logger.info(
            "Cache summary: lookups=%s full_hits=%s partial_hits=%s miss=%s full_hit_rate=%.2f%% any_hit_rate=%.2f%%",
            cache_stats_data.get("lookups", 0),
            cache_stats_data.get("full_hits", 0),
            cache_stats_data.get("partial_hits", 0),
            cache_stats_data.get("miss", 0),
            float(cache_stats_data.get("full_hit_rate_pct", 0.0)),
            float(cache_stats_data.get("any_hit_rate_pct", 0.0)),
        )
        buffer_stats_data = self.stats.get("buffer", {})
        sample_stats = buffer_stats_data.get("sample_buffer", {})
        judging_stats = buffer_stats_data.get("judging_buffer", {})
        logger.info(
            "Buffer summary: sample mean_batch=%.2f batches=%s max_queue=%s | judging mean_batch=%.2f batches=%s max_queue=%s",
            float(sample_stats.get("mean_batch_size", 0.0)),
            sample_stats.get("batches", 0),
            sample_stats.get("max_queue_size", 0),
            float(judging_stats.get("mean_batch_size", 0.0)),
            judging_stats.get("batches", 0),
            judging_stats.get("max_queue_size", 0),
        )
        logger.info("Diagnostics: %s", diagnostics)
        if refusal_filter_stats:
            logger.info(
                "Refusal filter summary: refusal=%s no_refusal_checked=%s reverted=%s total=%s | refusal_ratio=%.2f%% no_refusal_ratio=%.2f%% reverted_ratio=%.2f%%",
                refusal_filter_stats.get("refusal_total", refusal_filter_stats.get("confirmed_refusal", 0)),
                refusal_filter_stats.get("no_refusal_checked", 0),
                refusal_filter_stats.get("reverted_total", refusal_filter_stats.get("reverted_after_transition", 0)),
                refusal_filter_stats.get("total", refusal_filter_stats.get("total_checked", 0)),
                float(refusal_filter_stats.get("confirmed_ratio_pct", 0.0)),
                float(refusal_filter_stats.get("no_refusal_checked_ratio_pct", 0.0)),
                float(refusal_filter_stats.get("reverted_ratio_pct", 0.0)),
            )
    
    
    
    def apply_chat_template(self, prompt: str):
        """Format prompt using target model's chat template and return (templated_text, token_ids)."""
        tokenizer = self.sampler.tokenizer
        model_name = self.config.target_model or ""
        original_prompt = prompt
        

        tools = None
        chat = None
        try:
            payload = json.loads(original_prompt)
            if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
                chat = payload["messages"]
                maybe_tools = payload.get("tools")
                if isinstance(maybe_tools, list):
                    tools = maybe_tools
        except (json.JSONDecodeError, KeyError, TypeError):
            chat = None

        if chat is None and "gemma-3" in model_name.lower():
            chat = [{"role": "user", "content": [{"type": "text", "text": original_prompt}]}]
        elif chat is None:
            chat = [{"role": "user", "content": original_prompt}]

        # Apply chat template (only if tokenizer supports it - not in Tokenizer interface)
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                if "Qwen3-8B" in model_name:
                    prompt_with_chat_template = tokenizer.apply_chat_template(
                        chat, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    logger.info(f"Prompt: {prompt_with_chat_template}")
                else:
                    prompt_with_chat_template = tokenizer.apply_chat_template(
                        chat, tools=tools, tokenize=False, add_generation_prompt=True
                    )
                # assistant_prefill_ids = [
                #     198, 40, 4157, 1855, 264, 707,309
                # ]
                # prompt_with_chat_template_ids = tokenizer.encode(prompt_with_chat_template) + assistant_prefill_ids
                # prompt_with_chat_template = prompt_with_chat_template + tokenizer.decode(assistant_prefill_ids)
                # return prompt_with_chat_template, prompt_with_chat_template_ids
                logger.info(f"Prompt with chat template: {prompt_with_chat_template}")
                return prompt_with_chat_template, tokenizer.encode(prompt_with_chat_template)
            else:
                # Tokenizer doesn't support chat template, return original prompt
                logger.info(
                    f"the tokenizer does not support chat template, return original prompt: {original_prompt}"
                )
                if not original_prompt.endswith(". "):
                    original_prompt = original_prompt + ". "
                return original_prompt, tokenizer.encode(original_prompt)
        except Exception as e:
            logger.error(f"Error formatting prompt with chat template: {e}")
            if not original_prompt.endswith(". "):
                original_prompt = original_prompt + ". "
            return original_prompt, tokenizer.encode(original_prompt)
