import math
from typing import List, Tuple

import numpy as np

from ..models.truss import Truss


class TrussBuilder:
    """Factory utilities for common truss geometries used in this project."""

    @staticmethod
    def _grid_nodes(nx: int, ny: int, Lx: float, Ly: float, z: float) -> np.ndarray:
        xs = np.linspace(0.0, Lx, nx)
        ys = np.linspace(0.0, Ly, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.full_like(X, z)
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    @classmethod
    def build_double_layer_space_truss(
        cls,
        nx: int = 6,
        ny: int = 6,
        Lx: float = 10.0,
        Ly: float = 10.0,
        gap: float = 1.0,
        A_top: float = 2.0e-4,
        A_bot: float = 2.0e-4,
        A_diag: float = 1.5e-4,
        E_all: float = 210e9,
        z0: float = 0.0,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]:
        top = cls._grid_nodes(nx, ny, Lx, Ly, z0 + gap / 2.0)
        bot = cls._grid_nodes(nx, ny, Lx, Ly, z0 - gap / 2.0)

        nodes = np.vstack([top, bot])

        def idx_top(i: int, j: int) -> int:
            return i * ny + j

        def idx_bot(i: int, j: int) -> int:
            return nx * ny + i * ny + j

        edges: List[Tuple[int, int]] = []
        A_list: List[float] = []
        E_list: List[float] = []

        for layer_fn, A_val in ((idx_top, A_top), (idx_bot, A_bot)):
            for i in range(nx):
                for j in range(ny):
                    u = layer_fn(i, j)
                    if i + 1 < nx:
                        v = layer_fn(i + 1, j)
                        edges.append((u, v))
                        A_list.append(A_val)
                        E_list.append(E_all)
                    if j + 1 < ny:
                        v = layer_fn(i, j + 1)
                        edges.append((u, v))
                        A_list.append(A_val)
                        E_list.append(E_all)
                    if i + 1 < nx and j + 1 < ny:
                        v1 = layer_fn(i + 1, j + 1)
                        edges.append((u, v1))
                        A_list.append(A_val)
                        E_list.append(E_all)

                        v2 = layer_fn(i + 1, j)
                        w2 = layer_fn(i, j + 1)
                        edges.append((v2, w2))
                        A_list.append(A_val)
                        E_list.append(E_all)

        for i in range(nx):
            for j in range(ny):
                t = idx_top(i, j)
                for di in (0, 1):
                    for dj in (0, 1):
                        ii, jj = i + di, j + dj
                        if ii < nx and jj < ny:
                            b = idx_bot(ii, jj)
                            edges.append((t, b))
                            A_list.append(A_diag)
                            E_list.append(E_all)

        return (
            nodes,
            edges,
            np.asarray(A_list, dtype=np.float64),
            np.asarray(E_list, dtype=np.float64),
        )

    @staticmethod
    def build_tower_truss(
        n_levels: int = 8,
        n_sides: int = 4,
        radius: float = 2.0,
        story: float = 2.0,
        A_vert: float = 2.0e-4,
        A_diag: float = 1.5e-4,
        A_ring: float = 1.5e-4,
        E_all: float = 210e9,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]:
        nodes = []
        for k in range(n_levels + 1):
            z = k * story
            for s in range(n_sides):
                ang = 2.0 * math.pi * s / n_sides
                nodes.append([radius * math.cos(ang), radius * math.sin(ang), z])
        nodes = np.asarray(nodes, dtype=np.float64)

        def idx(k: int, s: int) -> int:
            return k * n_sides + (s % n_sides)

        edges: List[Tuple[int, int]] = []
        A_list: List[float] = []
        E_list: List[float] = []

        for k in range(n_levels + 1):
            for s in range(n_sides):
                edges.append((idx(k, s), idx(k, s + 1)))
                A_list.append(A_ring)
                E_list.append(E_all)

        for k in range(n_levels):
            for s in range(n_sides):
                edges.append((idx(k, s), idx(k + 1, s)))
                A_list.append(A_vert)
                E_list.append(E_all)

                edges.append((idx(k, s), idx(k + 1, s + 1)))
                A_list.append(A_diag)
                E_list.append(E_all)

                edges.append((idx(k, s + 1), idx(k + 1, s)))
                A_list.append(A_diag)
                E_list.append(E_all)

        return (
            nodes,
            edges,
            np.asarray(A_list, dtype=np.float64),
            np.asarray(E_list, dtype=np.float64),
        )

    @classmethod
    def make_dlt_problem(
        cls,
        nx: int = 6,
        ny: int = 2,
        Lx: float = 12.0,
        Ly: float = 4.0,
        gap: float = 1.2,
        load_vec: np.ndarray = np.array([0.0, 0.0, -1200.0]),
        top_only: bool = True,
        tol: float = 1e-9,
    ) -> Truss:
        nodes, edges, A, E = cls.build_double_layer_space_truss(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, gap=gap
        )

        loads = np.zeros((nodes.shape[0], 3), dtype=np.float64)
        fixed = np.zeros((nodes.shape[0], 3), dtype=bool)

        x = nodes[:, 0]
        z = nodes[:, 2]
        xmin, xmax = float(x.min()), float(x.max())
        zmed = float(np.median(z))
        is_top = z >= (zmed + 1e-12)

        left_idx = np.where(np.abs(x - xmin) <= tol)[0]
        fixed[left_idx, :] = True

        right_idx = np.where(np.abs(x - xmax) <= tol)[0]
        if top_only:
            right_idx = right_idx[is_top[right_idx]]

        for i in right_idx:
            loads[i] += load_vec

        tr = Truss(nodes=nodes, edges=edges, A=A, E=E, loads=loads, fixed=fixed)
        tr.validate()
        return tr


# Backward-compatible function API.
def build_double_layer_space_truss(*args, **kwargs):
    return TrussBuilder.build_double_layer_space_truss(*args, **kwargs)


def build_tower_truss(*args, **kwargs):
    return TrussBuilder.build_tower_truss(*args, **kwargs)


def make_dlt_problem(*args, **kwargs):
    return TrussBuilder.make_dlt_problem(*args, **kwargs)
