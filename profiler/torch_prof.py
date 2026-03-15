"""
maybe_torch_profile: wraps a block with torch.profiler and saves a Chrome
trace to output_dir/torch_profile.json.

Usage (automatic via config.torch_profiler_steps > 0):
    set_torch_profile_dir("results/my_run")
    # sampler will call maybe_torch_profile internally

Manual usage:
    from profiler.torch_prof import maybe_torch_profile
    with maybe_torch_profile("results/debug", steps=5):
        for _ in range(5):
            sampler.cs.sample_step(logits, sf)
            prof.step()

View the trace at: https://ui.perfetto.dev  (open torch_profile.json)
"""

import contextlib
import os


@contextlib.contextmanager
def maybe_torch_profile(output_dir: str, steps: int = 5):
    """
    Profile GPU kernels for *steps* active steps and save Chrome trace.
    Yields the profiler object so callers can call prof.step() per iteration.
    If torch.profiler is unavailable, yields None (no-op).
    """
    if not output_dir or steps <= 0:
        yield None
        return

    try:
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule
    except ImportError:
        yield None
        return

    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "torch_profile.json")

    sched = schedule(wait=1, warmup=1, active=steps, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        yield prof

    prof.export_chrome_trace(trace_path)
    summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    summary_path = os.path.join(output_dir, "torch_profile_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"[torch_profiler] trace -> {trace_path}")
    print(f"[torch_profiler] summary -> {summary_path}")
    print(summary)
