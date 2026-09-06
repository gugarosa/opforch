# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Configure OPForch's stdout and delayed rotating-file logging.

"""  # fmt: skip

from __future__ import annotations

import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "opforch.log"


def get_console_handler() -> StreamHandler:
    """Create a stdout handler with the shared OPForch formatter.

    The handler is not attached to a logger, and closing it does not close stdout.

    Returns:
        A new stream handler bound to the current sys.stdout.

    """

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    return handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Create a delayed file handler that rotates at local midnight.

    Records append to LOG_FILE relative to the current working directory, with no backup-count limit.
    The file is opened on the first emitted record, not when the handler is created.
    The returned handler is unattached and should be closed when it is no longer needed.

    Returns:
        A new midnight-rotating file handler using the shared OPForch formatter.

    Raises:
        OSError: Existing log-file metadata cannot be inspected during handler initialization.

    """

    handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    handler.setFormatter(FORMATTER)
    return handler


def get_logger(logger_name: str) -> Logger:
    """Return a named logger without duplicating its handlers.

    A logger without directly attached handlers receives DEBUG level, stdout and delayed file handlers,
    and disabled propagation. Existing handlers, levels, and propagation settings are left unchanged.
    Repeated calls return the same standard-library logger.

    Args:
        logger_name: Name passed to logging.getLogger, normally the calling module's __name__.

    Returns:
        The cached logger, configured only if it has no directly attached handlers.

    Raises:
        TypeError: The logger name is not accepted by logging.getLogger.
        OSError: Existing log-file metadata cannot be inspected when creating the file handler.

    """

    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
        logger.propagate = False

    return logger
