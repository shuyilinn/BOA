# profiler

Lightweight function timing module. Zero overhead when disabled. Supports inclusive/exclusive time. Results saved automatically to `results/`.

---

## Core Concepts

| Concept | Description |
|---|---|
| `ProfileSession` | A named session that accumulates timing data |
| `set_session` / `clear_session` | Activate/deactivate the session on the current thread (call once at entry point) |
| `@profile` | Decorator — add to a function definition once, every call is timed automatically |
| `profile_block` | Context manager — for one-off blocks where a decorator isn't practical |
| `total_sec` | **Inclusive**: wall time including nested profiled sub-calls |
| `self_sec` | **Exclusive**: wall time minus time spent in nested profiled sub-calls |

---

## Usage 1: Integrated in a run (already set up)

`executor.py` is already wired. Every `executor.run()` call profiles `sampler.flush_once` and `judger.flush_once` and saves results to:

```
results/{run_id}/
├── profile.json   # structured data
└── profile.txt    # human-readable table
```

Just run as usual:

```bash
python run.py
```

---

## Usage 2: Add timing to any function

Add `@profile` once to the function definition — no other changes needed anywhere:

```python
from profiler import profile

@profile("sampler.expand")       # custom name
def expand(self, node):
    ...

@profile                          # defaults to module.qualname
def my_function():
    ...
```

Zero overhead when no session is active.

---

## Usage 3: Standalone analysis of a single file/module

To profile a module in isolation without running the full experiment:

```python
from profiler import ProfileSession, set_session, clear_session, profile

# Step 1: decorate the functions you care about (one-time)
@profile("sampler.batch_generate")
def batch_generate(...):
    ...

# Step 2: activate a session before calling them
session = ProfileSession(name="my_test")
set_session(session)

batch_generate(...)   # called normally — timed automatically
batch_generate(...)

clear_session()

# Step 3: view or save results
print(session.summary_text())
session.save("results/my_test")   # writes profile.json + profile.txt
```

---

## Usage 4: profile_block (for one-off code blocks)

```python
from profiler import profile_block

with profile_block("expander.expand"):
    new_nodes = expander.expand(node)
```

Requires an active session, otherwise a no-op.

---

## Inclusive vs Exclusive Example

If A calls B and both are decorated with `@profile`:

```
A: total_sec=1.0s, self_sec=0.3s   ← A's own logic took 0.3s
B: total_sec=0.7s, self_sec=0.7s   ← B has no nested profiled calls
```

Use `total_sec` to see "how much wall time does this function own end-to-end".
Use `self_sec` to find where time is actually being spent.

---

## Output Example

`profile.txt`:

```
Profile session: 20260312-131059_...
-------------------------------------------------------------------------------------
Function                             Calls  Total(s)   Self(s)  Mean(s)  Min(s)  Max(s)
-------------------------------------------------------------------------------------
sampler.flush_once                      42     28.30     28.30    0.674   0.510   1.230
judger.flush_once                       38     10.20     10.20    0.268   0.190   0.450
-------------------------------------------------------------------------------------
```
