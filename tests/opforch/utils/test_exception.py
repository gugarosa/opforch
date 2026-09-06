# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest

from opforch.utils import exception


def test_error_preserves_custom_exception_class_and_message():
    message = "`value` is invalid."
    error = exception.Error("Error", message)

    with pytest.raises(exception.Error) as caught:
        raise error

    assert caught.value is error
    assert str(error) == message


@pytest.mark.parametrize(
    "error_class",
    [exception.ArgumentError, exception.BuildError, exception.SizeError, exception.TypeError, exception.ValueError],
)
def test_exception_subclasses_preserve_base_class_identity_and_message(error_class):
    message = "`value` is invalid."
    error = error_class(message)

    with pytest.raises(error_class) as caught:
        raise error

    assert caught.value is error
    assert isinstance(error, exception.Error)
    assert str(error) == message
