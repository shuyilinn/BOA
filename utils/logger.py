import logging
from pathlib import Path
"""
Shuyi: This file is for logging the experiment.
TODO: check the logging file redirection.

"""

_DEFAULT_LOG_FILE = "logs/out.log"
_DEFAULT_LOG_MODE = "debug"


def set_default_log_file(log_file: str) -> None:
    global _DEFAULT_LOG_FILE
    _DEFAULT_LOG_FILE = log_file


def set_default_log_mode(mode: str) -> None:
    global _DEFAULT_LOG_MODE
    _DEFAULT_LOG_MODE = str(mode).strip().lower()


def setup_logger(
    name: str,
    mode: str | None = None,
    log_to_file: bool = True,
    log_file: str | None = None,
):
    """
    mode:
      - debug   : DEBUG and above
      - info    : INFO and above
      - warning : WARNING and above
      - error   : ERROR and above
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    resolved_mode = str(mode if mode is not None else _DEFAULT_LOG_MODE).strip().lower()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
    )

    if resolved_mode == "debug":
        stream_level = logging.DEBUG

    elif resolved_mode == "info":
        stream_level = logging.INFO

    elif resolved_mode == "warning":
        stream_level = logging.WARNING

    elif resolved_mode == "error":
        stream_level = logging.ERROR

    else:
        raise ValueError(f"Unknown mode: {resolved_mode}")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_to_file:
        log_path = Path(log_file or _DEFAULT_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(stream_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
