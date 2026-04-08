[acceleration]
L0: users: depending [multi-turn]
L1: inter chunk: use heuristic search
L2: inner chunk : beam search or [vllm prefix accelerator]

engines:
sampler: (customized sampler)
1. vllm [may not work]
2. hf

expander:
1. vllm [prefix]
2. hf []

Note: hf and vllm has different sampler strategy, so use it consistently!

should call the engines in the engines dir

