"""OPForch exception types."""

from __future__ import annotations

from opforch.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """Base OPForch exception."""

    def __init__(self, cls: str, msg: str) -> None:
        super().__init__(f"{cls}: {msg}")
        logger.error("%s: %s.", cls, msg)


class ArgumentError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("ArgumentError", error)


class BuildError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("BuildError", error)


class SizeError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("SizeError", error)


class TypeError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("TypeError", error)


class ValueError(Error):
    def __init__(self, error: str) -> None:
        super().__init__("ValueError", error)
