"""Optimum-Path Forest abstract base classifier."""

from typing import List, Optional

import torch

import opforch.math.distance as d
import opforch.utils.exception as e
from opforch.utils import logging
from opforch.utils.device import DeviceManager

logger = logging.get_logger(__name__)


class OPF:
    """Abstract base class defining all common OPF-related methods.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki.
        LibOPF: A library for the design of optimum-path forest classifiers (2015).

    """

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            distance: Name of the distance metric to use.
            pre_computed_distance: Path to a pre-computed distance file.
            device: Target device string ('cpu', 'cuda', 'cuda:0', etc.).

        """

        logger.info("Creating class: OPF.")

        self.device = DeviceManager.resolve(device)

        self.subgraph = None

        if distance not in d.VALID_DISTANCES:
            raise e.TypeError(
                f"`distance` should be one of: {', '.join(sorted(d.VALID_DISTANCES))}"
            )

        self.distance = distance
        self.distance_fn = d.DISTANCES[distance]

        if pre_computed_distance:
            self.pre_computed_distance = True
            self._read_distances(pre_computed_distance)
        else:
            self.pre_computed_distance = False
            self.pre_distances = None

        logger.debug(
            "Distance: %s | Pre-computed distance: %s | Device: %s.",
            self.distance,
            self.pre_computed_distance,
            self.device,
        )
        logger.info("Class created.")

    def _read_distances(self, file_name: str) -> None:
        """Reads pre-computed distances from a file.

        Args:
            file_name: Path to the distance file (.csv, .txt, or .pt).

        """

        logger.debug("Running private method: read_distances().")

        extension = file_name.split(".")[-1]

        if extension in ("pt", "pth"):
            distances = torch.load(
                file_name, map_location=self.device, weights_only=True
            )
        elif extension == "csv":
            import numpy as np

            data = np.loadtxt(file_name, delimiter=",")
            distances = torch.from_numpy(data).to(
                dtype=torch.float64, device=self.device
            )
        elif extension == "txt":
            import numpy as np

            data = np.loadtxt(file_name, delimiter=" ")
            distances = torch.from_numpy(data).to(
                dtype=torch.float64, device=self.device
            )
        else:
            raise e.ArgumentError(
                "File extension not recognized. "
                "It should be `.csv`, `.txt`, `.pt` or `.pth`"
            )

        if distances is None:
            raise e.ValueError("Pre-computed distances could not be properly loaded")

        self.pre_distances = distances

    def get_distances(self, normalize: bool = False) -> torch.Tensor:
        """Computes the full distance matrix for the training subgraph.

        Args:
            normalize: Whether to min-max normalize the matrix. Constant matrices
                normalize to zeros.

        Returns:
            Distance matrix of shape (N, N).

        """

        distances = self._get_distances()

        if normalize:
            d_min = distances.min()
            d_max = distances.max()
            scale = d_max - d_min
            return (distances - d_min) / torch.where(scale == 0, 1, scale)

        return distances

    def _get_distances(
        self,
        X: Optional[torch.Tensor] = None,
        I: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.subgraph is None:
            raise e.BuildError("Subgraph has not been properly created")
        if self.pre_computed_distance and self.pre_distances is None:
            raise e.BuildError("Pre-computed distances have not been loaded")
        return self.subgraph._get_distances(
            self.distance_fn,
            self.pre_distances if self.pre_computed_distance else None,
            X,
            I,
        )

    def to(self, device) -> "OPF":
        """Moves the entire model to a device.

        Args:
            device: Target device string or torch.device.

        Returns:
            Self, for chaining.

        """

        self.device = torch.device(device) if isinstance(device, str) else device

        if self.subgraph is not None:
            self.subgraph.to(self.device)

        if self.pre_distances is not None:
            self.pre_distances = self.pre_distances.to(self.device)

        return self

    def save(self, file_name: str) -> None:
        """Saves the model using torch.save.

        Args:
            file_name: File path to save to.

        """

        logger.info("Saving model to file: %s ...", file_name)

        # Move to CPU before saving for portability
        cpu_model = self.to("cpu")
        torch.save(cpu_model, file_name)

        logger.info("Model saved.")

    def load(self, file_name: str) -> None:
        """Loads a model from a file.

        Args:
            file_name: File path to load from.

        """

        logger.info("Loading model from file: %s ...", file_name)

        loaded = torch.load(file_name, map_location=self.device, weights_only=False)
        self.__dict__.update(loaded.__dict__)

        logger.info("Model loaded.")

    def fit(self, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> None:
        """Fits data in the classifier.

        Must be implemented by subclasses.

        Args:
            X: Feature tensor of shape (N, D).
            Y: Label tensor of shape (N,).

        """

        raise NotImplementedError

    def predict(self, X: torch.Tensor, **kwargs) -> List[int]:
        """Predicts new data using the pre-trained classifier.

        Must be implemented by subclasses.

        Args:
            X: Feature tensor of shape (M, D).

        Returns:
            A list of predicted labels.

        """

        raise NotImplementedError
