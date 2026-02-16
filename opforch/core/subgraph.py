"""Tensor-first Subgraph for the Optimum-Path Forest.

Replaces the OPFython Node + Subgraph pair with dense tensors.
All per-node state lives as columns in this class, enabling
batch operations and seamless GPU transfer.
"""

from typing import Optional, Tuple

import numpy as np
import torch

import opforch.stream.parser as p
import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.stream import loader
from opforch.utils import logging
from opforch.utils.device import DeviceManager

logger = logging.get_logger(__name__)


class Subgraph:
    """A tensor-based collection of nodes forming the OPF subgraph.

    All per-node attributes are stored as dense 1-D or 2-D tensors,
    enabling vectorized operations and device transfer.
    """

    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        I: Optional[torch.Tensor] = None,
        from_file: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            X: Feature array/tensor of shape (N, D).
            Y: Label array/tensor of shape (N,).
            I: Index array/tensor of shape (N,).
            from_file: Path to load data from (.csv, .txt, or .json).
            device: Target device string (e.g. 'cpu', 'cuda').

        """

        self.device = DeviceManager.resolve(device)
        self.trained = False

        if from_file:
            X, Y = self._load(from_file)

        if X is not None:
            self._build(X, Y, I)
        else:
            # Empty subgraph — tensors will be initialized later
            self.features = torch.empty(0, device=self.device)
            self.labels = torch.empty(0, dtype=torch.int64, device=self.device)
            self._n_features = 0
            self._init_state_tensors(0)
            logger.error("Subgraph has not been properly created.")

    def _to_tensor(self, arr, dtype=torch.float32) -> torch.Tensor:
        """Converts numpy arrays, lists, or existing tensors to the target device."""

        if arr is None:
            return None

        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
        elif isinstance(arr, torch.Tensor):
            t = arr
        else:
            t = torch.tensor(arr)

        return t.to(dtype=dtype, device=self.device)

    def _load(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and parses a dataframe from a file.

        Args:
            file_path: File to be loaded.

        Returns:
            Tuple of (features, labels) tensors.

        """

        extension = file_path.split(".")[-1]

        if extension == "csv":
            data = loader.load_csv(file_path)
        elif extension == "txt":
            data = loader.load_txt(file_path)
        elif extension == "json":
            data = loader.load_json(file_path)
        else:
            raise e.ArgumentError(
                "File extension not recognized. It should be `.csv`, `.json` or `.txt`"
            )

        X, Y = p.parse_loader(data)

        return X, Y

    def _init_state_tensors(self, n: int) -> None:
        """Initializes all per-node state tensors to default values.

        Args:
            n: Number of nodes.

        """

        dev = self.device

        self.pred_labels = torch.zeros(n, dtype=torch.int64, device=dev)
        self.cluster_labels = torch.zeros(n, dtype=torch.int64, device=dev)
        self.costs = torch.zeros(n, dtype=torch.float64, device=dev)
        self.densities = torch.zeros(n, dtype=torch.float64, device=dev)
        self.radii = torch.zeros(n, dtype=torch.float64, device=dev)
        self.n_plateaus = torch.zeros(n, dtype=torch.int64, device=dev)
        self.preds = torch.full((n,), c.NIL, dtype=torch.int64, device=dev)
        self.roots = torch.arange(n, dtype=torch.int64, device=dev)
        self.status = torch.full((n,), c.STANDARD, dtype=torch.int8, device=dev)
        self.relevant = torch.full((n,), c.IRRELEVANT, dtype=torch.int8, device=dev)

        # Ordered node indices (filled during training)
        self.idx_nodes = []

        # Adjacency (set by KNNSubgraph or during plateau expansion)
        self.adjacency = None

    def _build(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor],
        I: Optional[torch.Tensor],
    ) -> None:
        """Builds the subgraph from feature/label/index data.

        Args:
            X: Features of shape (N, D).
            Y: Labels of shape (N,). Defaults to zeros if None.
            I: Original indices of shape (N,). Defaults to 0..N-1 if None.

        """

        self.features = self._to_tensor(X, dtype=torch.float32)

        n = self.features.shape[0]

        if Y is not None:
            self.labels = self._to_tensor(Y, dtype=torch.int64)
        else:
            self.labels = torch.zeros(n, dtype=torch.int64, device=self.device)

        if I is not None:
            self.indices = self._to_tensor(I, dtype=torch.int64)
        else:
            self.indices = torch.arange(n, dtype=torch.int64, device=self.device)

        self._n_features = self.features.shape[1] if self.features.dim() > 1 else 0

        self._init_state_tensors(n)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the subgraph."""

        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        """Dimensionality of the feature space."""

        return self._n_features

    def to(self, device) -> "Subgraph":
        """Moves all tensors to the specified device.

        Args:
            device: Target torch device or string.

        Returns:
            Self, for chaining.

        """

        device = torch.device(device) if isinstance(device, str) else device
        self.device = device

        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.indices = self.indices.to(device)
        self.pred_labels = self.pred_labels.to(device)
        self.cluster_labels = self.cluster_labels.to(device)
        self.costs = self.costs.to(device)
        self.densities = self.densities.to(device)
        self.radii = self.radii.to(device)
        self.n_plateaus = self.n_plateaus.to(device)
        self.preds = self.preds.to(device)
        self.roots = self.roots.to(device)
        self.status = self.status.to(device)
        self.relevant = self.relevant.to(device)

        if self.adjacency is not None:
            self.adjacency = self.adjacency.to(device)

        return self

    def destroy_arcs(self) -> None:
        """Destroys all adjacency arcs in the subgraph."""

        self.n_plateaus.zero_()
        self.adjacency = None

    def mark_nodes(self, i: int) -> None:
        """Marks a node and its entire predecessor chain as relevant.

        Args:
            i: Starting node index.

        """

        while self.preds[i].item() != c.NIL:
            self.relevant[i] = c.RELEVANT
            i = self.preds[i].item()

        self.relevant[i] = c.RELEVANT

    def reset(self) -> None:
        """Resets predecessors, relevance flags, and arcs."""

        self.preds.fill_(c.NIL)
        self.relevant.fill_(c.IRRELEVANT)
        self.destroy_arcs()
