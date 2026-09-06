# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opforch.utils import logging


def test_get_console_handler_returns_handler():
    handler = logging.get_console_handler()

    assert handler is not None
    handler.close()


def test_get_timed_file_handler_returns_handler():
    handler = logging.get_timed_file_handler()

    assert handler is not None
    handler.close()


def test_get_logger_preserves_name_and_reuses_handlers():
    logger = logging.get_logger(__name__)
    handlers = len(logger.handlers)

    assert logger.name == __name__
    assert logger.hasHandlers() is True
    assert len(logging.get_logger(__name__).handlers) == handlers
