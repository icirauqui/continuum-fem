from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fea.solid import LinearSolidSolver
from .solid_builder import SolidBuilder


@dataclass(frozen=True, slots=True)
class TubeC3D6ForceResponse:
    """Outer-surface displacement response for one C3D6 tube point force."""

    s_norm: np.ndarray
    theta_norm: np.ndarray
    displacement: np.ndarray
    applied_node: int
    applied_axial_index: int
    applied_theta_index: int
    applied_s_norm: float
    applied_theta_norm: float
    force_vector: np.ndarray


def solve_hollow_cylinder_c3d6_force_response(
    *,
    length: float,
    inner_radius: float,
    outer_radius: float,
    n_axial: int,
    n_theta: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    center_s: float,
    center_theta: float,
    force_vector: np.ndarray,
) -> TubeC3D6ForceResponse:
    """Solve a fixed-end hollow C3D6 cylinder and return outer node displacements.

    The mesh follows SolidBuilder.build_hollow_cylinder_c3d6: axial coordinate is
    x, cross-section coordinates are y/z, and the radial layer index 1 is outer.
    Loads are applied to the nearest non-fixed outer node so fixed end reactions do
    not silently absorb a user-selected endpoint force.
    """

    force = np.asarray(force_vector, dtype=np.float64).reshape(3)
    solid = SolidBuilder.build_hollow_cylinder_c3d6(
        length=float(length),
        inner_radius=float(inner_radius),
        outer_radius=float(outer_radius),
        n_axial=int(n_axial),
        n_theta=int(n_theta),
        E=float(young_modulus_pa),
        nu=float(poisson_ratio),
        fix_ends=True,
    )

    axial_count = int(n_axial)
    theta_count = int(n_theta)
    if axial_count < 2:
        raise ValueError("n_axial must be >= 2")
    if theta_count < 6:
        raise ValueError("n_theta must be >= 6")

    axial_idx = int(round(float(np.clip(center_s, 0.0, 1.0)) * axial_count))
    axial_idx = int(np.clip(axial_idx, 1, axial_count - 1))
    theta_idx = int(round((float(center_theta) % 1.0) * theta_count)) % theta_count

    def node_id(i: int, j: int, radial_idx: int) -> int:
        return ((i * theta_count + (j % theta_count)) * 2) + radial_idx

    applied_node = node_id(axial_idx, theta_idx, 1)
    loads = np.zeros_like(solid.loads)
    loads[applied_node, :] = force
    solid.loads = loads
    solid.validate()

    result = LinearSolidSolver.solve(solid)
    displacement = np.zeros((axial_count + 1, theta_count, 3), dtype=np.float64)
    for i in range(axial_count + 1):
        for j in range(theta_count):
            displacement[i, j, :] = result.u[node_id(i, j, 1)]

    s_norm = np.linspace(0.0, 1.0, axial_count + 1, dtype=np.float64)
    theta_norm = np.arange(theta_count, dtype=np.float64) / float(theta_count)
    return TubeC3D6ForceResponse(
        s_norm=s_norm,
        theta_norm=theta_norm,
        displacement=displacement,
        applied_node=int(applied_node),
        applied_axial_index=int(axial_idx),
        applied_theta_index=int(theta_idx),
        applied_s_norm=float(s_norm[axial_idx]),
        applied_theta_norm=float(theta_norm[theta_idx]),
        force_vector=force,
    )
