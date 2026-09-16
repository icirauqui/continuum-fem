from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..models.truss import Truss


@dataclass(slots=True)
class FEAResult:
    u: np.ndarray
    reactions: np.ndarray
    elem_axial: np.ndarray
    elem_stress: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "u": self.u,
            "reactions": self.reactions,
            "elem_axial": self.elem_axial,
            "elem_stress": self.elem_stress,
        }


class LinearTrussSolver:
    """Linear static 3D truss FE solver (small displacement, axial bars)."""

    @staticmethod
    def elem_dir_and_len(xi: np.ndarray, xj: np.ndarray) -> Tuple[np.ndarray, float]:
        d = xj - xi
        L = float(np.linalg.norm(d))
        if L <= 0.0:
            raise ValueError("Zero-length element encountered")
        return d / L, L

    @staticmethod
    def elem_stiff_3d_truss(A: float, E: float, t: np.ndarray, L: float) -> np.ndarray:
        l, m, n = t
        c = np.array(
            [[l * l, l * m, l * n], [m * l, m * m, m * n], [n * l, n * m, n * n]],
            dtype=np.float64,
        )
        ke_3x3 = (A * E / L) * c
        return np.block([[ke_3x3, -ke_3x3], [-ke_3x3, ke_3x3]])

    @classmethod
    def assemble_global_K(
        cls, nodes: np.ndarray, edges: List[Tuple[int, int]], A: np.ndarray, E: np.ndarray
    ) -> np.ndarray:
        n_nodes = nodes.shape[0]
        K = np.zeros((3 * n_nodes, 3 * n_nodes), dtype=np.float64)

        for eidx, (i, j) in enumerate(edges):
            t, L = cls.elem_dir_and_len(nodes[i], nodes[j])
            ke = cls.elem_stiff_3d_truss(A[eidx], E[eidx], t, L)
            dofs = np.r_[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            K[np.ix_(dofs, dofs)] += ke
        return K

    @classmethod
    def solve(cls, truss: Truss) -> FEAResult:
        truss.validate()

        n_nodes = truss.n_nodes
        K = cls.assemble_global_K(truss.nodes, truss.edges, truss.A, truss.E)
        F = truss.loads.reshape(-1).astype(np.float64)

        fixed_mask = truss.fixed.reshape(-1)
        free = np.where(~fixed_mask)[0]
        if free.size == 0:
            raise ValueError("All DOFs are fixed; no free variables to solve")

        Kff = K[np.ix_(free, free)]
        Ff = F[free]
        uf = np.linalg.solve(Kff, Ff)

        u = np.zeros(3 * n_nodes, dtype=np.float64)
        u[free] = uf
        u = u.reshape(n_nodes, 3)

        reactions = (K @ u.reshape(-1) - F).reshape(n_nodes, 3)

        axial = np.zeros(truss.n_elements, dtype=np.float64)
        stress = np.zeros(truss.n_elements, dtype=np.float64)
        for eidx, (i, j) in enumerate(truss.edges):
            t, L = cls.elem_dir_and_len(truss.nodes[i], truss.nodes[j])
            ext = float(np.dot(t, (u[j] - u[i])))
            f = (truss.A[eidx] * truss.E[eidx] / L) * ext
            axial[eidx] = f
            stress[eidx] = f / truss.A[eidx]

        return FEAResult(u=u, reactions=reactions, elem_axial=axial, elem_stress=stress)


# Backward-compatible function API.
def _elem_dir_and_len(xi: np.ndarray, xj: np.ndarray) -> Tuple[np.ndarray, float]:
    return LinearTrussSolver.elem_dir_and_len(xi, xj)


def _elem_stiff_3d_truss(A: float, E: float, t: np.ndarray, L: float) -> np.ndarray:
    return LinearTrussSolver.elem_stiff_3d_truss(A, E, t, L)


def assemble_global_K(nodes: np.ndarray, edges: List[Tuple[int, int]], A: np.ndarray, E: np.ndarray) -> np.ndarray:
    return LinearTrussSolver.assemble_global_K(nodes, edges, A, E)


def solve_truss(truss: Truss) -> Dict[str, np.ndarray]:
    return LinearTrussSolver.solve(truss).as_dict()
