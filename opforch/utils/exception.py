# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide OPForch exceptions that log diagnostics when constructed.

All custom exceptions inherit from Error rather than similarly named built-in exception types.

"""

from __future__ import annotations

from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class Error(Exception):
    """Represent the base exception for OPForch failures.

    """  # fmt: skip

    def __init__(self, cls: str, msg: str) -> None:
        """Create an exception with unchanged message text and log its category.

        Args:
            cls: Exception category included in the diagnostic log but not prefixed to the exception text.
            msg: Message retained as the exception text and included in the diagnostic log.

        """

        super().__init__(msg)
        logger.error("`exception=%s` was reported (%s).", cls, msg)


class ArgumentError(Error):
    """Represent an invalid or missing argument configuration.

    """  # fmt: skip

    def __init__(self, error: str) -> None:
        """Create and log an argument error.

        Args:
            error: Message retained as the exception text and included in the diagnostic log.

        """

        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Represent an unavailable or invalid built model state.

    """  # fmt: skip

    def __init__(self, error: str) -> None:
        """Create and log a build-state error.

        Args:
            error: Message retained as the exception text and included in the diagnostic log.

        """

        super().__init__("BuildError", error)


class SizeError(Error):
    """Represent incompatible tensor or collection sizes.

    """  # fmt: skip

    def __init__(self, error: str) -> None:
        """Create and log a size error.

        Args:
            error: Message retained as the exception text and included in the diagnostic log.

        """

        super().__init__("SizeError", error)


class TypeError(Error):
    """Represent a value with an unsupported type.

    """  # fmt: skip

    def __init__(self, error: str) -> None:
        """Create and log a type error.

        Args:
            error: Message retained as the exception text and included in the diagnostic log.

        """

        super().__init__("TypeError", error)


class ValueError(Error):
    """Represent a value outside its accepted domain.

    """  # fmt: skip

    def __init__(self, error: str) -> None:
        """Create and log a value error.

        Args:
            error: Message retained as the exception text and included in the diagnostic log.

        """

        super().__init__("ValueError", error)
