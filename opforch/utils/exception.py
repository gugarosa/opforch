"""Custom exception hierarchy for OPForch."""

from opforch.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """A generic Error class derived from Exception.

    Logs the error class and message upon construction.

    """

    def __init__(self, cls: str, msg: str) -> None:
        """Initialization method.

        Args:
            cls: Class identifier.
            msg: Message to be logged.

        """

        super(Error, self).__init__(f"{cls}: {msg}")

        logger.error("%s: %s.", cls, msg)


class ArgumentError(Error):
    """Error for wrong number of provided arguments."""

    def __init__(self, error: str) -> None:
        super(ArgumentError, self).__init__("ArgumentError", error)


class BuildError(Error):
    """Error for classes not being properly built before use."""

    def __init__(self, error: str) -> None:
        super(BuildError, self).__init__("BuildError", error)


class SizeError(Error):
    """Error for wrong length or size of variables."""

    def __init__(self, error: str) -> None:
        super(SizeError, self).__init__("SizeError", error)


class TypeError(Error):
    """Error for wrong type of variables."""

    def __init__(self, error: str) -> None:
        super(TypeError, self).__init__("TypeError", error)


class ValueError(Error):
    """Error for wrong value of variables."""

    def __init__(self, error: str) -> None:
        super(ValueError, self).__init__("ValueError", error)
