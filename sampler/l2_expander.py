from typing import List, Optional

from boa_types.interaction import EnvironmentRequestBundle, META_ENDS_WITH_EOS
from boa_types.tree_node import NodeSource, TreeNode

from .l1_expander import L1Expander
from .l3_expander import L3Expander


class L2Expander:
    """
    Chunk Connector.
    Responsibility: call L3 to obtain path pieces, then materialize them as TreeNodes and attach them to the main tree.

    L2 only materializes structure (children content/topology/generation metrics).
    Do not set status/score(s); those are owned by Executor/Judger. See CONVENTIONS.md.
    """

    def __init__(
        self,
        engine, 
        config,
        threshold: Optional[List[float]] = None,
    ):
        self.l3_handler = L3Expander(engine, config, threshold=threshold)
        self.l1_expander = L1Expander(engine, config)
        self.tokenizer = engine.get_tokenizer()
        self._eos_token_ids: set[int] = {int(self.tokenizer.eos_token_id)}
        model = getattr(engine, "model", None)
        if model is not None:
            generation_eos = model.generation_config.eos_token_id
            if isinstance(generation_eos, int):
                self._eos_token_ids.add(generation_eos)
            else:
                self._eos_token_ids.update(int(t) for t in generation_eos)

    def expand(self, node: TreeNode) -> List[TreeNode]:
        """
        Call the black-box L3 and convert returned chunk sequences into child nodes on the tree.
        """
        prompt_len = int(node.prompt_len)
        base_generated_len = max(0, len(node.get_path_token_ids()) - prompt_len)

        candidate_chunks = self.l3_handler.find_candidate_chunks(
            node.get_path_token_ids(),
            base_cum_log_prob=float(node.cum_log_prob),
            base_generated_len=base_generated_len,
        )

        chunk_ids_batch = [chunk["ids"] for chunk in candidate_chunks]
        if hasattr(self.tokenizer, "batch_decode"):
            chunk_texts = self.tokenizer.batch_decode(chunk_ids_batch, skip_special_tokens=False)
        else:
            chunk_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in chunk_ids_batch]

        new_nodes = []
        for chunk, chunk_text in zip(candidate_chunks, chunk_texts):
            chunk_ids = chunk["ids"]
            ends_with_eos = bool(
                self._eos_token_ids
                and chunk_ids
                and int(chunk_ids[-1]) in self._eos_token_ids
            )
            child = node.add_child(
                token_ids=chunk_ids,
                text=chunk_text,
                log_prob=chunk["log_p"],
                **{META_ENDS_WITH_EOS: ends_with_eos},
                is_cut=bool(chunk.get("is_cut")),
                cut_reason=chunk.get("cut_reason"),
                cut_threshold=chunk.get("cut_threshold"),
                cut_th_idx=chunk.get("cut_th_idx"),
                cut_global_cum_log_prob=chunk.get("global_log_p"),
            )
            child.source = NodeSource.ASSISTANT
            if ends_with_eos:
                request_bundle = EnvironmentRequestBundle(
                    assistant_node=child,
                    env_type=child.environment_type,
                    metadata=dict(child.metadata),
                )
                new_nodes.append(self.l1_expander.expand_after_eos(request_bundle))
            else:
                new_nodes.append(child)

        return new_nodes
