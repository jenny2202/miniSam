# minisam.py
# --------------------------------------------------------------------------- #
# Lightweight Self-Organizing Map (online + batch training).                  #
# --------------------------------------------------------------------------- #

import logging  
import numpy as np
import math
from warnings import warn
from typing import Tuple, Optional


class MiniSam:

    #---------------------------- constructor -------------------------- #
    def __init__(
        self,
        x: int,
        y: int,
        input_len: int,
        sigma: Optional[float] = None,
        learning_rate: float = 0.5,
        neighborhood_function: str = "gaussian",
        random_seed: Optional[int] = None,
    ):
        """
        Initializes a Self Organizing Map.
        
        Parameters
            ----------
            x : int
                x dimension of the SOM.
            y : int
                y dimension of the SOM.
            input_len : int
                Number of the elements of the vectors in input.
            sigma : float or None, optional
                Initial neighbourhood radius σ₀.  Defaults to ``max(x, y) / 2``.
            learning_rate : float, default=0.5
                Initial learning rate.
            neighborhood_function : {"gaussian", "bubble"}, default="gaussian"
                Shape of the neighbourhood kernel.  *Gaussian* is the classic choice;
                *bubble* uses a hard cutoff within radius σ.
            random_seed : int or None, optional
                Seed for deterministic weight initialisation.
        """

        # Geometry & hyper-parameters
        self._x, self._y, self._dim = x, y, input_len
        self._sigma0 = sigma if sigma is not None else max(x, y) / 2.0
        
        # warning if sigma exceeds map diagonal
        if self._sigma0 > math.hypot(x, y):
            warn('Warning: sigma might be too high for the dimension of the map.')

        self._learning_rate = learning_rate
        self._random_generator = np.random.default_rng(random_seed)

        # Weight grid (x, y, dim)
        self._weight = self._random_generator.random((x, y, input_len))

        # Coordinate grid for distance computation 
        xs, ys = np.meshgrid(np.arange(x), np.arange(y), indexing="ij")
        self._coords = np.stack((xs, ys), axis=-1)

        # Neighbourhood kernel
        if neighborhood_function == "gaussian":
            self._kernel = self._gaussian
        elif neighborhood_function == "bubble":
            self._kernel = self._bubble
        else:
            raise ValueError(
                "Unknown neighborhood_function "
                f"{neighborhood_function!r} (choose 'gaussian' or 'bubble')."
            )

    # ---------------------------- initialization -------------------------- #
    def random_weights_init(self, data: np.ndarray) -> None:
        """Initializes the weights of the SOM
        picking random samples from data.
        """
        self._validate_data(data, "random_weights_init(data)")
        idx = self._random_generator.choice(len(data), self._x * self._y, replace=True)
        self._weight = data[idx].reshape(self._x, self._y, self._dim)

    # ------------------------------ online training --------------------------- #
    def train_random(self, data: np.ndarray, num_iteration: int) -> None:
        """
        Classic online training (one random sample per update).
        The learning rate and neighbourhood decay exponentially.
        """
        self._validate_data(data, "train_random(data)")
        if num_iteration <= 0:
            raise ValueError("num_iteration must be positive")
        
        lam = num_iteration / max(np.log(self._sigma0), 1e-12)

        for t in range(num_iteration):
            v = data[self._random_generator.integers(len(data))]
            sigma_t = self._sigma0 * np.exp(-t / lam)
            alpha_t = self._learning_rate * np.exp(-t / lam)

            i, j = self.winner(v)
            theta = self._kernel(self._sq_grid_dist(i, j), sigma_t)[..., None]
            self._weight += alpha_t * theta * (v - self._weight)

    # ------------------------------ offline training --------------------------- #
    def train_batch(self, data: np.ndarray, num_iteration: int) -> None:
        """
        Batch training (one weight update per epoch).
        """
        self._validate_data(data, "train_batch(data)")
        if num_iteration <= 0:
            raise ValueError("num_iteration must be positive")
        
        lam = num_iteration / max(np.log(self._sigma0), 1e-12)
        EPS = 1e-12
        
        for t in range(num_iteration):
            sigma_t = self._sigma0 * np.exp(-t / lam)
            alpha_t = self._learning_rate * np.exp(-t / lam)
            num = np.zeros_like(self._weight)
            den = np.zeros((self._x, self._y, 1))

            for v in data:
                i, j = self.winner(v)
                theta = self._kernel(self._sq_grid_dist(i, j), sigma_t)[..., None]
                num += theta * v
                den += theta
            self._weight = (1 - alpha_t) * self._weight + alpha_t * num / np.clip(den, EPS, None)

    # ------------------------------ inference --------------------------- #
    def winner(self, v: np.ndarray) -> Tuple[int, int]:
        """
        Computes the coordinates of the winning neuron for the sample v.
        Return (row, col) index of the Best-Matching Unit for vector v.
        """
        self._validate_data(np.atleast_2d(v), "winner(v)")

        dist2 = np.sum((self._weight - v) ** 2, axis=2)
        return tuple(np.unravel_index(np.argmin(dist2), dist2.shape))

    def quantization(self, data: np.ndarray) -> np.ndarray:
        """
        Assigns a weights vector of the winning neuron to each sample in data. 
        Map every vector in data to its BMU’s prototype weight.
        """
        self._validate_data(data, "quantization(data)")
        return np.array([self._weight[self.winner(v)] for v in data])

    def activation_response(self, data: np.ndarray) -> np.ndarray:
        """
        Hit histogram: counts how many times each neuron wins across data.
        """
        self._validate_data(data, "activation_response(data)")
        hits = np.zeros((self._x, self._y))
        for v in data:
            i, j = self.winner(v)
            hits[i, j] += 1
        return hits

    def distance_map(self) -> np.ndarray:
        """
        Returns the distance map of the weights.
        Looking only at the four Von-Neumann neighbours for demo purpose only, 
        but can be enhanced later with other types such as hexagonal, 8 neighbour(Moore) etc..  
        """
        um = np.zeros((self._x, self._y))
        for i in range(self._x):
            for j in range(self._y):
                w_ij = self._weight[i, j]
                neigh = [
                    self._weight[a, b]
                    for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                    if 0 <= a < self._x and 0 <= b < self._y
                ]
                if neigh:
                    um[i, j] = np.mean(np.linalg.norm(w_ij - neigh, axis=1))
        return um

    def get_weights(self) -> np.ndarray:
        """
        Returns the weights of the neural network.
        """
        return self._weight

    # -------------------------- Helper utilities -------------------------- #
    def _validate_data(self, data: np.ndarray, name: str = "data") -> None:
        """
        Verify that *data* is a finite NumPy array shaped (N, self._dim).

        Raises
        ------
        TypeError  – if data is not a NumPy ndarray  
        ValueError – if dimensionality or values are invalid
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, got {type(data).__name__}")

        if data.ndim != 2:
            raise ValueError(f"{name} must be 2-D (N, dim); got ndim={data.ndim}")

        if data.shape[1] != self._dim:
            raise ValueError(
                f"{name} second dimension mismatch: expected {self._dim}, got {data.shape[1]}"
            )

        if not np.isfinite(data).all():
            raise ValueError(f"{name} contains NaN or ±Inf values")

    def _sq_grid_dist(self, cx: int, cy: int) -> np.ndarray:
        diff = self._coords - self._coords[cx, cy]
        return diff[..., 0] ** 2 + diff[..., 1] ** 2

    @staticmethod
    def _gaussian(dist2: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-dist2 / (2.0 * sigma * sigma))

    @staticmethod
    def _bubble(dist2: np.ndarray, sigma: float) -> np.ndarray:
        return (dist2 <= sigma * sigma).astype(np.float64)
