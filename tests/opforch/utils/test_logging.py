from opforch.utils import logging


def test_get_console_handler():
    c = logging.get_console_handler()

    assert c is not None


def test_get_file_handler():
    f = logging.get_timed_file_handler()

    assert f is not None


def test_get_logger():
    logger = logging.get_logger(__name__)
    handlers = len(logger.handlers)

    assert logger.name == __name__
    assert logger.hasHandlers() is True
    assert len(logging.get_logger(__name__).handlers) == handlers
