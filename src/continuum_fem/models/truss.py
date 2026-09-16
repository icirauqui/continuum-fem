from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(slots=True)
class Truss:
    """3D truss model with per-node loads and DOF-wise support constraints."""

    nodes: np.ndarray
    edges: List[Tuple[int, int]]
    A: np.ndarray
    E: np.ndarray
    loads: np.ndarray
    fixed: np.ndarray

    def validate(self) -> None:
        """Validate shape and consistency assumptions for solver safety."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (N, 3)")
        n_nodes = self.nodes.shape[0]

        if self.loads.shape != (n_nodes, 3):
            raise ValueError("loads must have shape (N, 3)")
        if self.fixed.shape != (n_nodes, 3):
            raise ValueError("fixed must have shape (N, 3)")

        n_edges = len(self.edges)
        if self.A.shape != (n_edges,):
            raise ValueError("A must have shape (E,)")
        if self.E.shape != (n_edges,):
            raise ValueError("E must have shape (E,)")

        for eidx, (i, j) in enumerate(self.edges):
            if i == j:
                raise ValueError(f"edge {eidx} has repeated node index {i}")
            if not (0 <= i < n_nodes and 0 <= j < n_nodes):
                raise ValueError(f"edge {eidx} has out-of-range node index")

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(len(self.edges))
