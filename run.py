"""
This script is the entry point for the BOA experiment.
1. load the config, parse the command line arguments, and apply the overrides
2. TODO: future: load checkpoint and resume from the checkpoint
3. run the experiments, call executor to run the prompts one by one.
"""

from utils.config_resolver import resolve_config
from utils.logger import set_default_log_file, set_default_log_mode, setup_logger
from utils.run_naming import build_log_file_path, build_run_id


def main():
    cfg = resolve_config()
    import torch
    torch.cuda.set_device(cfg.target_model_cuda_number)
    run_id = build_run_id(cfg)
    setattr(cfg, "run_id", run_id)
    log_file = build_log_file_path(cfg, run_id=run_id)
    set_default_log_file(log_file)
    logger_mode = cfg.logger_mode
    set_default_log_mode(logger_mode)

    # Defer imports so module-level loggers pick up the run-specific log file.
    from benchmark.prompts_loader import load_benign_prompts, load_harmful_prompts
    from executor.executor import Executor
    from probability_threshold.threshold_builder import (
        build_or_load_threshold,
        try_load_threshold,
    )

    logger = setup_logger("run", mode=logger_mode)
    logger.warning(f"Log file: {log_file}")
    logger.info(f"Config: {cfg.to_dict()}")

    test_mode = bool(cfg.test_mode)

    # Load prompts (both loaders return {"samples": [{"prompt", ...}, ...]})
    benign_n = cfg.benign_n
    skip_benign = test_mode or benign_n == 0
    if skip_benign:
        benign_prompts = []
        logger.info("Skip benign dataset loading and threshold baseline (test_mode=%s, benign_n=%s).", test_mode, benign_n)
    else:
        benign_data = load_benign_prompts(dataset_name=cfg.benign_path, random_seed=cfg.random_seed)
        benign_prompts = benign_data["samples"][:benign_n] if benign_n else benign_data["samples"]
        logger.info("Loaded %s benign prompts from %s", len(benign_prompts), cfg.benign_path)

    workload_config = cfg.workload_configs[cfg.workload_name]
    harmful_path = workload_config["benchmark_path"]
    harmful_data = load_harmful_prompts(harmful_path)
    harmful_samples = harmful_data["samples"]
    start = int(cfg.harmful_prompt_start)
    end = cfg.harmful_prompt_end
    end_idx = None if end is None or int(end) == -1 else int(end)
    harmful_samples = harmful_samples[start - 1 : end_idx]
    harmful_prompts = [s["prompt"] for s in harmful_samples]
    original_prompts = [s["original_prompt"] for s in harmful_samples]
    logger.info(
        "Loaded %s harmful prompts and %s original prompts from %s (range=%s:%s)",
        len(harmful_prompts),
        len(original_prompts),
        harmful_path,
        start,
        "end" if end_idx is None else end_idx,
    )

    if len(harmful_prompts) == 0:
        logger.warning("No harmful prompts found, skipping executor initialization and experiment")
        return

    target_gpu = cfg.target_model_cuda_number
    judger_gpu = cfg.judger_cuda_number
    is_single_gpu = target_gpu == judger_gpu
    logger.info("GPU topology: single=%s (target=%s, judger=%s)", is_single_gpu, target_gpu, judger_gpu)

    occupancy_plan = "Occ1(target-only)" if is_single_gpu else "Occ2(target+judger)"
    logger.info("Occupancy plan: %s", occupancy_plan)

    if is_single_gpu:
        executor = Executor(cfg, threshold=None, init_target=True, init_judger=False)
    else:
        executor = Executor(cfg, threshold=None, init_target=True, init_judger=True)

    need_threshold = (not test_mode) and (len(benign_prompts) > 0)
    threshold = None
    threshold_loaded = False
    threshold_built = False

    if need_threshold:
        threshold = try_load_threshold(cfg)
        if threshold is not None:
            threshold_loaded = True
        else:
            benign_prompts_for_threshold = [s["prompt"] for s in benign_prompts]
            threshold = build_or_load_threshold(
                cfg,
                benign_prompts_for_threshold,
                engine=executor.target_model_engine,
            )
            threshold_built = True

    executor.set_threshold(threshold)
    logger.info("Threshold: need=%s, loaded=%s, built=%s", need_threshold, threshold_loaded, threshold_built)

    if is_single_gpu:
        import gc
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Late judger init for single-GPU mode.")
        executor.initialize_judger_components()

    total_prompts = len(harmful_prompts)
    for prompt_index, sample in enumerate(
        harmful_samples,
        start=1,
    ):
        harmful_prompt = sample["prompt"]
        original_prompt = sample["original_prompt"]
        prompt_metadata = {
            "environments": sample.get("environments", []),
            "tool_schemas": sample.get("tool_schemas", []),
            "tools_openai": sample.get("tools_openai", []),
            "fulfillable": sample.get("fulfillable"),
        }
        logger.info(f"Start running prompt: {harmful_prompt}")
        executor.run(
            harmful_prompt,
            original_prompt,
            prompt_metadata=prompt_metadata,
            prompt_index=prompt_index,
            total_prompts=total_prompts,
        )
    logger.info("Experiment finished")


if __name__ == "__main__":
    main()
