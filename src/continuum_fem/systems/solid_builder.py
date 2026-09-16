from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..models.solid import Solid


@dataclass(slots=True)
class _VoxelGeometry:
    nodes: np.ndarray
    hex_elements: np.ndarray
    hex_columns: np.ndarray


class SolidBuilder:
    """Factory utilities for solid benchmark geometries."""

    _HEX_EDGES = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    @staticmethod
    def _node_index(i: int, j: int, k: int, ny: int, nz: int) -> int:
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    @classmethod
    def _build_active_mask(
        cls,
        nx: int,
        ny: int,
        nz: int,
        h: float,
        a: float,
        b: float,
        c: float,
        p: float,
    ) -> np.ndarray:
        xc = (np.arange(nx, dtype=np.float64) + 0.5 - 0.5 * nx) * h
        yc = (np.arange(ny, dtype=np.float64) + 0.5 - 0.5 * ny) * h
        zc = (np.arange(nz, dtype=np.float64) + 0.5 - 0.5 * nz) * h

        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        val = (np.abs(X / max(a, 1e-12)) ** p) + (np.abs(Y / max(b, 1e-12)) ** p) + (np.abs(Z / max(c, 1e-12)) ** p)
        return val <= 1.0

    @classmethod
    def _largest_connected_component(cls, mask: np.ndarray) -> np.ndarray:
        nx, ny, nz = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best: List[Tuple[int, int, int]] = []

        neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not mask[i, j, k] or visited[i, j, k]:
                        continue
                    comp: List[Tuple[int, int, int]] = []
                    q: deque[Tuple[int, int, int]] = deque([(i, j, k)])
                    visited[i, j, k] = True
                    while q:
                        ci, cj, ck = q.popleft()
                        comp.append((ci, cj, ck))
                        for di, dj, dk in neigh:
                            ni, nj, nk = ci + di, cj + dj, ck + dk
                            if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz:
                                continue
                            if visited[ni, nj, nk] or not mask[ni, nj, nk]:
                                continue
                            visited[ni, nj, nk] = True
                            q.append((ni, nj, nk))
                    if len(comp) > len(best):
                        best = comp

        out = np.zeros_like(mask, dtype=bool)
        for i, j, k in best:
            out[i, j, k] = True
        return out

    @classmethod
    def _build_voxel_geometry(
        cls,
        nx: int,
        ny: int,
        nz: int,
        h: float,
        active: np.ndarray,
    ) -> _VoxelGeometry:
        xs = (np.arange(nx + 1, dtype=np.float64) - 0.5 * nx) * h
        ys = (np.arange(ny + 1, dtype=np.float64) - 0.5 * ny) * h
        zs = (np.arange(nz + 1, dtype=np.float64) - 0.5 * nz) * h
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

        hex_elems: List[List[int]] = []
        hex_cols: List[List[int]] = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not active[i, j, k]:
                        continue
                    n0 = cls._node_index(i, j, k, ny, nz)
                    n1 = cls._node_index(i + 1, j, k, ny, nz)
                    n2 = cls._node_index(i + 1, j + 1, k, ny, nz)
                    n3 = cls._node_index(i, j + 1, k, ny, nz)
                    n4 = cls._node_index(i, j, k + 1, ny, nz)
                    n5 = cls._node_index(i + 1, j, k + 1, ny, nz)
                    n6 = cls._node_index(i + 1, j + 1, k + 1, ny, nz)
                    n7 = cls._node_index(i, j + 1, k + 1, ny, nz)
                    hex_elems.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    hex_cols.append([i, j])

        if not hex_elems:
            raise ValueError("Active shape has no elements")

        return _VoxelGeometry(
            nodes=nodes,
            hex_elements=np.asarray(hex_elems, dtype=np.int64),
            hex_columns=np.asarray(hex_cols, dtype=np.int64),
        )

    @classmethod
    def _build_voxel_geometry_from_axes(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        active: np.ndarray,
    ) -> _VoxelGeometry:
        nx, ny, nz = active.shape
        if xs.shape != (nx + 1,) or ys.shape != (ny + 1,) or zs.shape != (nz + 1,):
            raise ValueError("axis arrays must match active voxel dimensions")

        X, Y, Z = np.meshgrid(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(zs, dtype=np.float64),
            indexing="ij",
        )
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

        hex_elems: List[List[int]] = []
        hex_cols: List[List[int]] = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not active[i, j, k]:
                        continue
                    n0 = cls._node_index(i, j, k, ny, nz)
                    n1 = cls._node_index(i + 1, j, k, ny, nz)
                    n2 = cls._node_index(i + 1, j + 1, k, ny, nz)
                    n3 = cls._node_index(i, j + 1, k, ny, nz)
                    n4 = cls._node_index(i, j, k + 1, ny, nz)
                    n5 = cls._node_index(i + 1, j, k + 1, ny, nz)
                    n6 = cls._node_index(i + 1, j + 1, k + 1, ny, nz)
                    n7 = cls._node_index(i, j + 1, k + 1, ny, nz)
                    hex_elems.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    hex_cols.append([i, j])

        if not hex_elems:
            raise ValueError("Active shape has no elements")

        return _VoxelGeometry(
            nodes=nodes,
            hex_elements=np.asarray(hex_elems, dtype=np.int64),
            hex_columns=np.asarray(hex_cols, dtype=np.int64),
        )

    @classmethod
    def _prune_and_remap(
        cls,
        nodes: np.ndarray,
        elems8: np.ndarray,
        elems6: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        used = np.zeros(nodes.shape[0], dtype=bool)
        if elems8.size > 0:
            used[np.unique(elems8.ravel())] = True
        if elems6.size > 0:
            used[np.unique(elems6.ravel())] = True

        old_to_new = -np.ones(nodes.shape[0], dtype=np.int64)
        old_to_new[np.where(used)[0]] = np.arange(np.count_nonzero(used), dtype=np.int64)

        new_nodes = nodes[used]
        new_e8 = old_to_new[elems8] if elems8.size > 0 else np.zeros((0, 8), dtype=np.int64)
        new_e6 = old_to_new[elems6] if elems6.size > 0 else np.zeros((0, 6), dtype=np.int64)
        return new_nodes, new_e8, new_e6

    @staticmethod
    def _wedge_oriented_positive(nodes: np.ndarray, wedge: np.ndarray) -> np.ndarray:
        # Quick orientation check at centroid-like point.
        # Swap local triangle orientation if Jacobian is negative.
        r = 1.0 / 3.0
        s = 1.0 / 3.0
        t = 0.0

        dndr = np.array(
            [
                -0.5 * (1.0 - t),
                0.5 * (1.0 - t),
                0.0,
                -0.5 * (1.0 + t),
                0.5 * (1.0 + t),
                0.0,
            ],
            dtype=np.float64,
        )
        dnds = np.array(
            [
                -0.5 * (1.0 - t),
                0.0,
                0.5 * (1.0 - t),
                -0.5 * (1.0 + t),
                0.0,
                0.5 * (1.0 + t),
            ],
            dtype=np.float64,
        )
        dndt = np.array(
            [
                -0.5 * (1.0 - r - s),
                -0.5 * r,
                -0.5 * s,
                0.5 * (1.0 - r - s),
                0.5 * r,
                0.5 * s,
            ],
            dtype=np.float64,
        )

        x = nodes[wedge]
        j11 = float(np.dot(dndr, x[:, 0]))
        j12 = float(np.dot(dnds, x[:, 0]))
        j13 = float(np.dot(dndt, x[:, 0]))
        j21 = float(np.dot(dndr, x[:, 1]))
        j22 = float(np.dot(dnds, x[:, 1]))
        j23 = float(np.dot(dndt, x[:, 1]))
        j31 = float(np.dot(dndr, x[:, 2]))
        j32 = float(np.dot(dnds, x[:, 2]))
        j33 = float(np.dot(dndt, x[:, 2]))
        detj = np.linalg.det(np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]], dtype=np.float64))

        if detj > 0.0:
            return wedge

        out = wedge.copy()
        out[[1, 2]] = out[[2, 1]]
        out[[4, 5]] = out[[5, 4]]
        return out

    @classmethod
    def hex_column_split_to_wedges(
        cls,
        nodes: np.ndarray,
        elements_c3d8: np.ndarray,
        hex_columns: np.ndarray,
        split_fraction: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if elements_c3d8.shape[0] != hex_columns.shape[0]:
            raise ValueError("elements_c3d8 and hex_columns must have same row count")

        unique_cols = np.unique(hex_columns, axis=0)
        if unique_cols.shape[0] == 0:
            return np.zeros((0, 8), dtype=np.int64), np.zeros((0, 6), dtype=np.int64)

        n_split = int(round(float(split_fraction) * unique_cols.shape[0]))
        n_split = max(0, min(unique_cols.shape[0], n_split))

        split_set: set[Tuple[int, int]] = set()
        if n_split > 0:
            pick = rng.choice(unique_cols.shape[0], size=n_split, replace=False)
            for idx in np.atleast_1d(pick):
                ij = unique_cols[int(idx)]
                split_set.add((int(ij[0]), int(ij[1])))

        kept_hex: List[np.ndarray] = []
        wedges: List[np.ndarray] = []

        for eidx, hex_nodes in enumerate(elements_c3d8):
            ij = (int(hex_columns[eidx, 0]), int(hex_columns[eidx, 1]))
            if ij not in split_set:
                kept_hex.append(hex_nodes)
                continue

            n0, n1, n2, n3, n4, n5, n6, n7 = [int(v) for v in hex_nodes]
            w1 = np.asarray([n0, n1, n2, n4, n5, n6], dtype=np.int64)
            w2 = np.asarray([n0, n2, n3, n4, n6, n7], dtype=np.int64)
            wedges.append(cls._wedge_oriented_positive(nodes, w1))
            wedges.append(cls._wedge_oriented_positive(nodes, w2))

        out_hex = np.asarray(kept_hex, dtype=np.int64) if kept_hex else np.zeros((0, 8), dtype=np.int64)
        out_wedge = np.asarray(wedges, dtype=np.int64) if wedges else np.zeros((0, 6), dtype=np.int64)
        return out_hex, out_wedge

    @classmethod
    def build_superquadric_voxel_solid(
        cls,
        nx: int,
        ny: int,
        nz: int,
        h: float,
        a: float,
        b: float,
        c: float,
        p: float,
        split_fraction: float,
        E_base: float = 8e9,
        nu_base: float = 0.30,
        rng: np.random.Generator | None = None,
    ) -> Solid:
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("nx, ny, nz must be >= 2")
        if h <= 0.0:
            raise ValueError("h must be > 0")

        if rng is None:
            rng = np.random.default_rng()

        active = cls._build_active_mask(nx=nx, ny=ny, nz=nz, h=h, a=a, b=b, c=c, p=p)
        active = cls._largest_connected_component(active)
        if np.count_nonzero(active) == 0:
            raise ValueError("No active voxels in superquadric mask")

        geom = cls._build_voxel_geometry(nx=nx, ny=ny, nz=nz, h=h, active=active)
        elems8, elems6 = cls.hex_column_split_to_wedges(
            nodes=geom.nodes,
            elements_c3d8=geom.hex_elements,
            hex_columns=geom.hex_columns,
            split_fraction=split_fraction,
            rng=rng,
        )

        nodes, elems8, elems6 = cls._prune_and_remap(geom.nodes, elems8, elems6)

        E8 = np.full((elems8.shape[0],), float(E_base), dtype=np.float64)
        nu8 = np.full((elems8.shape[0],), float(nu_base), dtype=np.float64)
        E6 = np.full((elems6.shape[0],), float(E_base), dtype=np.float64)
        nu6 = np.full((elems6.shape[0],), float(nu_base), dtype=np.float64)

        solid = Solid(
            nodes=nodes,
            elements_c3d8=elems8,
            elements_c3d6=elems6,
            E_c3d8=E8,
            nu_c3d8=nu8,
            E_c3d6=E6,
            nu_c3d6=nu6,
            loads=np.zeros((nodes.shape[0], 3), dtype=np.float64),
            fixed=np.zeros((nodes.shape[0], 3), dtype=bool),
        )
        solid.validate()
        return solid

    @classmethod
    def build_parametric_ibeam_c3d8(
        cls,
        *,
        length: float,
        height: float,
        flange_width: float,
        flange_thickness: float,
        web_thickness: float,
        n_length: int = 10,
        n_width: int = 7,
        n_height: int = 7,
        E: float = 210.0e9,
        nu: float = 0.30,
        total_load: float = 25_000.0,
        load_direction: Sequence[float] = (0.0, 0.0, -1.0),
        load_patch_center_y: float = 0.0,
        load_patch_width_fraction: float = 0.55,
    ) -> Solid:
        """Build a clamped parametric I-beam using C3D8 voxels.

        The beam axis is x, flange width is y, and section height is z. The
        left x-face is fully fixed. The load is distributed over a patch on the
        right-end top flange, so the generated cases produce bending-dominated
        static responses while keeping the mesh small enough for CPU tests.
        """
        if length <= 0.0:
            raise ValueError("length must be > 0")
        if height <= 0.0:
            raise ValueError("height must be > 0")
        if flange_width <= 0.0:
            raise ValueError("flange_width must be > 0")
        if flange_thickness <= 0.0:
            raise ValueError("flange_thickness must be > 0")
        if web_thickness <= 0.0:
            raise ValueError("web_thickness must be > 0")
        if flange_thickness >= 0.45 * height:
            raise ValueError("flange_thickness must leave a web span")
        if web_thickness >= flange_width:
            raise ValueError("web_thickness must be smaller than flange_width")
        if n_length < 2 or n_width < 3 or n_height < 3:
            raise ValueError("n_length >= 2, n_width >= 3, and n_height >= 3 are required")
        if E <= 0.0:
            raise ValueError("E must be > 0")
        if not (-0.99 < nu < 0.49):
            raise ValueError("nu must be in (-0.99, 0.49)")
        if total_load <= 0.0:
            raise ValueError("total_load must be > 0")

        xs = np.linspace(0.0, float(length), int(n_length) + 1, dtype=np.float64)
        ys = np.linspace(-0.5 * float(flange_width), 0.5 * float(flange_width), int(n_width) + 1, dtype=np.float64)
        zs = np.linspace(-0.5 * float(height), 0.5 * float(height), int(n_height) + 1, dtype=np.float64)
        yc = 0.5 * (ys[:-1] + ys[1:])
        zc = 0.5 * (zs[:-1] + zs[1:])

        top_rows = zc >= (0.5 * float(height) - float(flange_thickness) - 1.0e-12)
        bottom_rows = zc <= (-0.5 * float(height) + float(flange_thickness) + 1.0e-12)
        if not bool(np.any(top_rows)):
            top_rows[int(np.argmax(zc))] = True
        if not bool(np.any(bottom_rows)):
            bottom_rows[int(np.argmin(zc))] = True

        web_cols = np.abs(yc) <= (0.5 * float(web_thickness) + 1.0e-12)
        if not bool(np.any(web_cols)):
            web_cols[int(np.argmin(np.abs(yc)))] = True

        cross_section = np.zeros((int(n_width), int(n_height)), dtype=bool)
        cross_section[:, top_rows] = True
        cross_section[:, bottom_rows] = True
        cross_section[web_cols, :] = True
        active = np.broadcast_to(cross_section.reshape(1, int(n_width), int(n_height)), (int(n_length), int(n_width), int(n_height))).copy()

        geom = cls._build_voxel_geometry_from_axes(xs=xs, ys=ys, zs=zs, active=active)
        nodes, elems8, elems6 = cls._prune_and_remap(
            geom.nodes,
            geom.hex_elements,
            np.zeros((0, 6), dtype=np.int64),
        )

        fixed = np.zeros((nodes.shape[0], 3), dtype=bool)
        tol_x = max(1.0e-10, 1.0e-9 * float(length))
        fixed[np.abs(nodes[:, 0]) <= tol_x, :] = True

        loads = np.zeros((nodes.shape[0], 3), dtype=np.float64)
        direction = np.asarray(load_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError("load_direction must be non-zero")
        direction = direction / norm

        x_max = float(np.max(nodes[:, 0]))
        z_max = float(np.max(nodes[:, 2]))
        y_half_patch = max(0.5 * float(load_patch_width_fraction) * float(flange_width), 1.0e-12)
        y_center = float(np.clip(load_patch_center_y, -0.45 * float(flange_width), 0.45 * float(flange_width)))
        loaded_nodes = np.where(
            (np.abs(nodes[:, 0] - x_max) <= tol_x)
            & (np.abs(nodes[:, 2] - z_max) <= max(1.0e-10, 1.0e-9 * float(height)))
            & (np.abs(nodes[:, 1] - y_center) <= y_half_patch)
        )[0]
        if loaded_nodes.size == 0:
            loaded_nodes = np.where(np.abs(nodes[:, 0] - x_max) <= tol_x)[0]
        if loaded_nodes.size == 0:
            raise ValueError("I-beam load patch selected no nodes")

        nodal_load = direction * (float(total_load) / float(loaded_nodes.size))
        loads[loaded_nodes, :] = nodal_load.reshape(1, 3)

        solid = Solid(
            nodes=nodes,
            elements_c3d8=elems8,
            elements_c3d6=elems6,
            E_c3d8=np.full((elems8.shape[0],), float(E), dtype=np.float64),
            nu_c3d8=np.full((elems8.shape[0],), float(nu), dtype=np.float64),
            E_c3d6=np.zeros((0,), dtype=np.float64),
            nu_c3d6=np.zeros((0,), dtype=np.float64),
            loads=loads,
            fixed=fixed,
        )
        solid.validate()
        return solid

    @classmethod
    def build_hollow_cylinder_c3d6(
        cls,
        length: float,
        inner_radius: float,
        outer_radius: float,
        n_axial: int,
        n_theta: int,
        E: float = 8e9,
        nu: float = 0.30,
        fix_ends: bool = True,
    ) -> Solid:
        """Build a single-layer hollow tube mesh using only C3D6 wedge elements."""
        if length <= 0.0:
            raise ValueError("length must be > 0")
        if inner_radius <= 0.0:
            raise ValueError("inner_radius must be > 0")
        if outer_radius <= inner_radius:
            raise ValueError("outer_radius must be > inner_radius")
        if n_axial < 2:
            raise ValueError("n_axial must be >= 2")
        if n_theta < 6:
            raise ValueError("n_theta must be >= 6")
        if E <= 0.0:
            raise ValueError("E must be > 0")
        if not (-0.99 < nu < 0.49):
            raise ValueError("nu must be in (-0.99, 0.49)")

        def node_id(i: int, j: int, radial_idx: int) -> int:
            return ((i * n_theta + (j % n_theta)) * 2) + radial_idx

        xs = np.linspace(0.0, float(length), n_axial + 1, dtype=np.float64)
        theta = (2.0 * np.pi * np.arange(n_theta, dtype=np.float64)) / float(n_theta)
        radii = np.asarray([inner_radius, outer_radius], dtype=np.float64)

        nodes: List[List[float]] = []
        for x in xs:
            for th in theta:
                c = float(np.cos(th))
                s = float(np.sin(th))
                for radius in radii:
                    nodes.append([float(x), float(radius * c), float(radius * s)])
        nodes_arr = np.asarray(nodes, dtype=np.float64)

        wedges: List[np.ndarray] = []
        for i in range(n_axial):
            for j in range(n_theta):
                jp = (j + 1) % n_theta
                n0 = node_id(i, j, 0)
                n1 = node_id(i + 1, j, 0)
                n2 = node_id(i + 1, jp, 0)
                n3 = node_id(i, jp, 0)
                n4 = node_id(i, j, 1)
                n5 = node_id(i + 1, j, 1)
                n6 = node_id(i + 1, jp, 1)
                n7 = node_id(i, jp, 1)

                w1 = np.asarray([n0, n1, n2, n4, n5, n6], dtype=np.int64)
                w2 = np.asarray([n0, n2, n3, n4, n6, n7], dtype=np.int64)
                wedges.append(cls._wedge_oriented_positive(nodes_arr, w1))
                wedges.append(cls._wedge_oriented_positive(nodes_arr, w2))

        elems6 = np.asarray(wedges, dtype=np.int64)
        elems8 = np.zeros((0, 8), dtype=np.int64)
        fixed = np.zeros((nodes_arr.shape[0], 3), dtype=bool)
        if fix_ends:
            fixed[: 2 * n_theta, :] = True
            fixed[-2 * n_theta :, :] = True

        solid = Solid(
            nodes=nodes_arr,
            elements_c3d8=elems8,
            elements_c3d6=elems6,
            E_c3d8=np.zeros((0,), dtype=np.float64),
            nu_c3d8=np.zeros((0,), dtype=np.float64),
            E_c3d6=np.full((elems6.shape[0],), float(E), dtype=np.float64),
            nu_c3d6=np.full((elems6.shape[0],), float(nu), dtype=np.float64),
            loads=np.zeros((nodes_arr.shape[0], 3), dtype=np.float64),
            fixed=fixed,
        )
        solid.validate()
        return solid


# Backward-compatible convenience exports
build_superquadric_voxel_solid = SolidBuilder.build_superquadric_voxel_solid
build_parametric_ibeam_c3d8 = SolidBuilder.build_parametric_ibeam_c3d8
build_hollow_cylinder_c3d6 = SolidBuilder.build_hollow_cylinder_c3d6
hex_column_split_to_wedges = SolidBuilder.hex_column_split_to_wedges
