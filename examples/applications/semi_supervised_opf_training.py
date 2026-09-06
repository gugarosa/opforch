# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Evaluate a semi-supervised forest using labeled and unlabeled boat samples.

Run this example from the repository root after installing OPForch.

"""

from pathlib import Path

import opforch.math.general as g
import opforch.stream.parser as p
import opforch.stream.splitter as s
from opforch.models.semi_supervised import SemiSupervisedOPF
from opforch.stream import loader

data_path = Path("data", "boat.txt")
txt = loader.load_txt(str(data_path))
if txt is None:
    raise OSError(f"`data_path` could not be loaded, but got {data_path}.")

X, Y = p.parse_loader(txt)
if X is None or Y is None:
    raise ValueError("`data` could not be parsed as an OPF table.")

X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.8, random_state=1)
X_train, X_unlabeled, Y_train, Y_unlabeled = s.split(X_train, Y_train, percentage=0.25, random_state=1)

opf = SemiSupervisedOPF(distance="log_squared_euclidean", device="cpu")
opf.fit(X_train, Y_train, X_unlabeled)

preds = opf.predict(X_test)
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
