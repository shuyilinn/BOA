"""
Thread-local active session registry.

set_session() / clear_session() control which ProfileSession is currently
active on this thread. Decorators and context managers read from here so
callers never need to pass a session argument around.
"""

import threading
from typing import Optional

_local = threading.local()


def set_session(session) -> None:
    """Activate *session* on the current thread."""
    _local.session = session


def get_session() -> Optional[object]:
    """Return the active session on this thread, or None."""
    return getattr(_local, "session", None)


def clear_session() -> None:
    """Deactivate the current session on this thread."""
    _local.session = None


def set_torch_profile_dir(path: Optional[str]) -> None:
    """Set the output directory for torch.profiler traces on this thread."""
    _local.torch_profile_dir = path


def get_torch_profile_dir() -> Optional[str]:
    """Return the torch.profiler output directory, or None if not set."""
    return getattr(_local, "torch_profile_dir", None)
