"""
@profile decorator and profile_block context manager.

Both read the active session from the thread-local registry (profiler.context),
so no session argument is ever needed at the call site.

Zero overhead when no session is active.
"""

import functools
from contextlib import contextmanager
from typing import Callable, Optional, Union

from profiler.context import get_session


def profile(name_or_fn: Union[str, Callable, None] = None) -> Callable:
    """
    Decorator. Can be used in three forms:

        @profile
        def flush_once(self, ...): ...

        @profile("sampler.flush")
        def flush_once(self, ...): ...

        flush_once = profile("sampler.flush")(flush_once)

    The profiled name defaults to ``module.qualname`` when not given.
    Uses whatever session is active on the current thread (set via
    set_session()). If no session is active, the function runs with zero
    overhead.
    """
    def _make_wrapper(fn: Callable, name: Optional[str]) -> Callable:
        _name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            session = get_session()
            if session is None or not session.enabled:
                return fn(*args, **kwargs)
            start = session._enter(_name)
            try:
                return fn(*args, **kwargs)
            finally:
                session._exit(_name, start)

        return wrapper

    if callable(name_or_fn):
        # @profile  (no parentheses)
        return _make_wrapper(name_or_fn, None)
    else:
        # @profile("custom.name")  or  profile("name")(fn)
        return lambda fn: _make_wrapper(fn, name_or_fn)


@contextmanager
def profile_block(name: str):
    """
    Context manager for one-off blocks that aren't worth decorating.

        with profile_block("expander.expand"):
            new_nodes = expander.expand(node)

    Uses the active session on the current thread. No-op if none is set.
    """
    session = get_session()
    if session is None or not session.enabled:
        yield
        return
    start = session._enter(name)
    try:
        yield
    finally:
        session._exit(name, start)
