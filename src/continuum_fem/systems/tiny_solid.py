from __future__ import annotations

import numpy as np

from ..models.solid import Solid
from .solid_builder import SolidBuilder


def unit_c3d8_nodes() -> np.ndarray:
    """Return C3D8 nodes in the local order used by the solid solver."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def unit_c3d6_nodes() -> np.ndarray:
    """Return a positive-orientation C3D6 triangular prism."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _empty_c3d8() -> np.ndarray:
    return np.zeros((0, 8), dtype=np.int64)


def _empty_c3d6() -> np.ndarray:
    return np.zeros((0, 6), dtype=np.int64)


def _material(count: int, value: float) -> np.ndarray:
    return np.full((int(count),), float(value), dtype=np.float64)


def make_single_c3d8_solid(
    *,
    E: float = 1000.0,
    nu: float = 0.30,
    total_load: float = 100.0,
    load_axis: int = 0,
) -> Solid:
    nodes = unit_c3d8_nodes()
    elements = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    fixed = np.zeros((nodes.shape[0], 3), dtype=bool)
    fixed[np.isclose(nodes[:, 0], 0.0), :] = True
    loads = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    loaded = np.where(np.isclose(nodes[:, 0], 1.0))[0]
    loads[loaded, int(load_axis)] = float(total_load) / float(max(1, loaded.size))
    return Solid(
        nodes=nodes,
        elements_c3d8=elements,
        elements_c3d6=_empty_c3d6(),
        E_c3d8=_material(1, E),
        nu_c3d8=_material(1, nu),
        E_c3d6=_material(0, E),
        nu_c3d6=_material(0, nu),
        loads=loads,
        fixed=fixed,
    )


def make_single_c3d6_solid(
    *,
    E: float = 1000.0,
    nu: float = 0.30,
    total_load: float = 100.0,
    load_axis: int = 2,
) -> Solid:
    nodes = unit_c3d6_nodes()
    elements = np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    fixed = np.zeros((nodes.shape[0], 3), dtype=bool)
    fixed[np.isclose(nodes[:, 2], 0.0), :] = True
    loads = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    loaded = np.where(np.isclose(nodes[:, 2], 1.0))[0]
    loads[loaded, int(load_axis)] = float(total_load) / float(max(1, loaded.size))
    return Solid(
        nodes=nodes,
        elements_c3d8=_empty_c3d8(),
        elements_c3d6=elements,
        E_c3d8=_material(0, E),
        nu_c3d8=_material(0, nu),
        E_c3d6=_material(1, E),
        nu_c3d6=_material(1, nu),
        loads=loads,
        fixed=fixed,
    )


def make_c3d8_cantilever_chain(
    n_elements: int,
    *,
    E: float = 1000.0,
    nu: float = 0.30,
    total_load: float = 100.0,
    load_axis: int = 0,
) -> Solid:
    if int(n_elements) < 1:
        raise ValueError("n_elements must be >= 1")
    nx = int(n_elements)
    nodes: list[list[float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for i in range(nx + 1):
        for j in range(2):
            for k in range(2):
                node_id[(i, j, k)] = len(nodes)
                nodes.append([float(i), float(j), float(k)])
    elements: list[list[int]] = []
    for i in range(nx):
        elements.append(
            [
                node_id[(i, 0, 0)],
                node_id[(i + 1, 0, 0)],
                node_id[(i + 1, 1, 0)],
                node_id[(i, 1, 0)],
                node_id[(i, 0, 1)],
                node_id[(i + 1, 0, 1)],
                node_id[(i + 1, 1, 1)],
                node_id[(i, 1, 1)],
            ]
        )
    node_arr = np.asarray(nodes, dtype=np.float64)
    fixed = np.zeros((node_arr.shape[0], 3), dtype=bool)
    fixed[np.isclose(node_arr[:, 0], 0.0), :] = True
    loads = np.zeros((node_arr.shape[0], 3), dtype=np.float64)
    loaded = np.where(np.isclose(node_arr[:, 0], float(nx)))[0]
    loads[loaded, int(load_axis)] = float(total_load) / float(max(1, loaded.size))
    return Solid(
        nodes=node_arr,
        elements_c3d8=np.asarray(elements, dtype=np.int64),
        elements_c3d6=_empty_c3d6(),
        E_c3d8=_material(nx, E),
        nu_c3d8=_material(nx, nu),
        E_c3d6=_material(0, E),
        nu_c3d6=_material(0, nu),
        loads=loads,
        fixed=fixed,
    )


def make_mixed_c3d8_c3d6_chain(
    n_cells: int,
    *,
    split_cells: tuple[int, ...] | None = None,
    E: float = 1000.0,
    nu: float = 0.30,
    total_load: float = 100.0,
    load_axis: int = 0,
    load_patch: str = "face",
) -> Solid:
    """Return a rectangular cantilever chain with selected bricks split into C3D6 wedges."""
    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1")
    nx = int(n_cells)
    split_set = set(int(v) for v in (split_cells if split_cells is not None else tuple(range(1, nx, 2))))
    invalid = [v for v in split_set if v < 0 or v >= nx]
    if invalid:
        raise ValueError(f"split_cells must be in [0, {nx}); got {invalid}")
    load_patch_norm = load_patch.strip().lower()
    if load_patch_norm not in {"face", "corner"}:
        raise ValueError("load_patch must be 'face' or 'corner'")

    nodes: list[list[float]] = []
    node_id: dict[tuple[int, int, int], int] = {}
    for i in range(nx + 1):
        for j in range(2):
            for k in range(2):
                node_id[(i, j, k)] = len(nodes)
                nodes.append([float(i), float(j), float(k)])
    node_arr = np.asarray(nodes, dtype=np.float64)

    hex_elements: list[list[int]] = []
    wedge_elements: list[np.ndarray] = []
    for i in range(nx):
        n0 = node_id[(i, 0, 0)]
        n1 = node_id[(i + 1, 0, 0)]
        n2 = node_id[(i + 1, 1, 0)]
        n3 = node_id[(i, 1, 0)]
        n4 = node_id[(i, 0, 1)]
        n5 = node_id[(i + 1, 0, 1)]
        n6 = node_id[(i + 1, 1, 1)]
        n7 = node_id[(i, 1, 1)]
        if i in split_set:
            w1 = np.asarray([n0, n1, n2, n4, n5, n6], dtype=np.int64)
            w2 = np.asarray([n0, n2, n3, n4, n6, n7], dtype=np.int64)
            wedge_elements.append(SolidBuilder._wedge_oriented_positive(node_arr, w1))
            wedge_elements.append(SolidBuilder._wedge_oriented_positive(node_arr, w2))
        else:
            hex_elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    elems8 = np.asarray(hex_elements, dtype=np.int64) if hex_elements else _empty_c3d8()
    elems6 = np.asarray(wedge_elements, dtype=np.int64) if wedge_elements else _empty_c3d6()
    fixed = np.zeros((node_arr.shape[0], 3), dtype=bool)
    fixed[np.isclose(node_arr[:, 0], 0.0), :] = True
    loads = np.zeros((node_arr.shape[0], 3), dtype=np.float64)
    if load_patch_norm == "face":
        loaded = np.where(np.isclose(node_arr[:, 0], float(nx)))[0]
    else:
        loaded = np.asarray([node_id[(nx, 1, 1)]], dtype=np.int64)
    loads[loaded, int(load_axis)] = float(total_load) / float(max(1, loaded.size))

    return Solid(
        nodes=node_arr,
        elements_c3d8=elems8,
        elements_c3d6=elems6,
        E_c3d8=_material(elems8.shape[0], E),
        nu_c3d8=_material(elems8.shape[0], nu),
        E_c3d6=_material(elems6.shape[0], E),
        nu_c3d6=_material(elems6.shape[0], nu),
        loads=loads,
        fixed=fixed,
    )
