from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Solid:
    """3D mixed solid model with C3D8 and C3D6 elements."""

    nodes: np.ndarray
    elements_c3d8: np.ndarray
    elements_c3d6: np.ndarray
    E_c3d8: np.ndarray
    nu_c3d8: np.ndarray
    E_c3d6: np.ndarray
    nu_c3d6: np.ndarray
    loads: np.ndarray
    fixed: np.ndarray
    prescribed_displacements: np.ndarray | None = None
    contact_candidate: np.ndarray | None = None
    contact_normal: np.ndarray | None = None
    contact_gap: np.ndarray | None = None
    contact_partner: np.ndarray | None = None
    contact_tangent_stick: np.ndarray | None = None
    contact_tangent_slip: np.ndarray | None = None
    contact_tangent_friction: np.ndarray | None = None
    contact_friction_mu: np.ndarray | None = None
    contact_incremental_steps: int | None = None
    contact_partner_remap: bool | None = None
    contact_dynamic_steps: int | None = None
    contact_dynamic_dt: float | None = None
    contact_dynamic_damping: float | None = None
    contact_dynamic_mass: np.ndarray | None = None
    contact_penalty_stiffness: np.ndarray | None = None
    contact_penalty_exponent: float | None = None

    def validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (N, 3)")
        n_nodes = int(self.nodes.shape[0])

        if self.loads.shape != (n_nodes, 3):
            raise ValueError("loads must have shape (N, 3)")
        if self.fixed.shape != (n_nodes, 3):
            raise ValueError("fixed must have shape (N, 3)")
        if self.fixed.dtype != bool:
            raise ValueError("fixed must be bool array")
        if self.prescribed_displacements is not None:
            if self.prescribed_displacements.shape != (n_nodes, 3):
                raise ValueError("prescribed_displacements must have shape (N, 3)")
            if np.any(~np.isfinite(self.prescribed_displacements)):
                raise ValueError("prescribed_displacements must be finite")
        if self.contact_incremental_steps is not None:
            if int(self.contact_incremental_steps) != self.contact_incremental_steps:
                raise ValueError("contact_incremental_steps must be an integer")
            if int(self.contact_incremental_steps) < 1:
                raise ValueError("contact_incremental_steps must be >= 1")
        dynamic_fields = (
            self.contact_dynamic_steps,
            self.contact_dynamic_dt,
            self.contact_dynamic_damping,
            self.contact_dynamic_mass,
        )
        if any(field is not None for field in dynamic_fields):
            if not all(field is not None for field in dynamic_fields):
                raise ValueError(
                    "contact_dynamic_steps, contact_dynamic_dt, contact_dynamic_damping, "
                    "and contact_dynamic_mass must be provided together"
                )
            if int(self.contact_dynamic_steps) != self.contact_dynamic_steps:
                raise ValueError("contact_dynamic_steps must be an integer")
            if int(self.contact_dynamic_steps) < 1:
                raise ValueError("contact_dynamic_steps must be >= 1")
            if not np.isfinite(float(self.contact_dynamic_dt)) or float(self.contact_dynamic_dt) <= 0.0:
                raise ValueError("contact_dynamic_dt must be finite and > 0")
            if not np.isfinite(float(self.contact_dynamic_damping)) or float(self.contact_dynamic_damping) < 0.0:
                raise ValueError("contact_dynamic_damping must be finite and >= 0")
            if self.contact_dynamic_mass.shape != (n_nodes, 3):
                raise ValueError("contact_dynamic_mass must have shape (N, 3)")
            if np.any(~np.isfinite(self.contact_dynamic_mass)):
                raise ValueError("contact_dynamic_mass must be finite")
            if np.any(self.contact_dynamic_mass <= 0.0):
                raise ValueError("contact_dynamic_mass must be positive")
        if self.contact_partner_remap is not None and not isinstance(
            self.contact_partner_remap,
            (bool, np.bool_),
        ):
            raise ValueError("contact_partner_remap must be bool")
        if self.contact_penalty_stiffness is not None or self.contact_penalty_exponent is not None:
            if self.contact_penalty_stiffness is None or self.contact_penalty_exponent is None:
                raise ValueError("contact_penalty_stiffness and contact_penalty_exponent must be provided together")
            if self.contact_penalty_stiffness.shape != (n_nodes,):
                raise ValueError("contact_penalty_stiffness must have shape (N,)")
            if np.any(~np.isfinite(self.contact_penalty_stiffness)):
                raise ValueError("contact_penalty_stiffness must be finite")
            if np.any(self.contact_penalty_stiffness < 0.0):
                raise ValueError("contact_penalty_stiffness must be nonnegative")
            if not np.isfinite(float(self.contact_penalty_exponent)) or float(self.contact_penalty_exponent) <= 1.0:
                raise ValueError("contact_penalty_exponent must be finite and > 1")
        contact_fields = (self.contact_candidate, self.contact_normal, self.contact_gap)
        if bool(self.contact_partner_remap) and not any(field is not None for field in contact_fields):
            raise ValueError("contact_partner_remap requires contact metadata")
        if self.contact_penalty_stiffness is not None and not any(field is not None for field in contact_fields):
            raise ValueError("contact_penalty_stiffness requires contact metadata")
        if (
            self.contact_tangent_stick is not None
            or self.contact_tangent_slip is not None
            or self.contact_tangent_friction is not None
            or self.contact_friction_mu is not None
        ) and not any(field is not None for field in contact_fields):
            raise ValueError("tangential contact fields require contact metadata")
        if any(field is not None for field in contact_fields):
            if not all(field is not None for field in contact_fields):
                raise ValueError("contact_candidate, contact_normal, and contact_gap must be provided together")
            if self.contact_candidate.shape != (n_nodes, 3):
                raise ValueError("contact_candidate must have shape (N, 3)")
            if self.contact_candidate.dtype != bool:
                raise ValueError("contact_candidate must be bool array")
            if self.contact_normal.shape != (n_nodes, 3):
                raise ValueError("contact_normal must have shape (N, 3)")
            if self.contact_gap.shape != (n_nodes, 3):
                raise ValueError("contact_gap must have shape (N, 3)")
            if np.any(~np.isfinite(self.contact_normal)) or np.any(~np.isfinite(self.contact_gap)):
                raise ValueError("contact_normal and contact_gap must be finite")
            if np.any(self.contact_gap[self.contact_candidate] < 0.0):
                raise ValueError("contact_gap must be nonnegative on contact candidates")
            if np.any(np.abs(self.contact_normal[self.contact_candidate]) <= 1.0e-12):
                raise ValueError("contact_normal must be nonzero on contact candidates")
            if self.contact_partner is not None:
                if self.contact_partner.shape != (n_nodes,):
                    raise ValueError("contact_partner must have shape (N,)")
                if not np.issubdtype(self.contact_partner.dtype, np.integer):
                    raise ValueError("contact_partner must be an integer array")
                if np.any((self.contact_partner < -1) | (self.contact_partner >= n_nodes)):
                    raise ValueError("contact_partner values must be -1 or valid node indices")
                candidate_nodes = np.any(self.contact_candidate, axis=1)
                if np.any(self.contact_partner[candidate_nodes] == np.where(candidate_nodes)[0]):
                    raise ValueError("contact_partner cannot reference the candidate node itself")
            if bool(self.contact_partner_remap):
                if self.contact_partner is None:
                    raise ValueError("contact_partner_remap requires contact_partner")
                candidate_nodes = np.any(self.contact_candidate, axis=1)
                if np.any(self.contact_partner[candidate_nodes] < 0):
                    raise ValueError("contact_partner_remap requires valid partners on contact candidates")
            if self.contact_penalty_stiffness is not None:
                candidate_nodes = np.any(self.contact_candidate, axis=1)
                if np.any(self.contact_penalty_stiffness[candidate_nodes] <= 0.0):
                    raise ValueError("contact_penalty_stiffness must be positive on contact candidates")
            if self.contact_tangent_stick is not None:
                if self.contact_tangent_stick.shape != (n_nodes, 3):
                    raise ValueError("contact_tangent_stick must have shape (N, 3)")
                if self.contact_tangent_stick.dtype != bool:
                    raise ValueError("contact_tangent_stick must be bool array")
                if self.contact_partner is None:
                    raise ValueError("contact_tangent_stick requires contact_partner")
                stick_nodes = np.any(self.contact_tangent_stick, axis=1)
                if np.any(~np.any(self.contact_candidate[stick_nodes], axis=1)):
                    raise ValueError("contact_tangent_stick nodes must also be contact candidates")
                if np.any(self.contact_partner[stick_nodes] < 0):
                    raise ValueError("contact_tangent_stick nodes must have valid partners")
                if np.any(self.contact_tangent_stick & self.contact_candidate):
                    raise ValueError("contact_tangent_stick components must be separate from normal candidates")
            if self.contact_friction_mu is not None:
                if self.contact_friction_mu.shape != (n_nodes,):
                    raise ValueError("contact_friction_mu must have shape (N,)")
                if np.any(~np.isfinite(self.contact_friction_mu)):
                    raise ValueError("contact_friction_mu must be finite")
                if np.any(self.contact_friction_mu < 0.0):
                    raise ValueError("contact_friction_mu must be nonnegative")
                if self.contact_tangent_slip is None and self.contact_tangent_friction is None:
                    raise ValueError("contact_friction_mu requires sliding or inferred-friction tangential metadata")

            if self.contact_tangent_slip is not None:
                if self.contact_friction_mu is None:
                    raise ValueError("contact_tangent_slip requires contact_friction_mu")
                if self.contact_tangent_slip.shape != (n_nodes, 3):
                    raise ValueError("contact_tangent_slip must have shape (N, 3)")
                if np.any(~np.isfinite(self.contact_tangent_slip)):
                    raise ValueError("contact_tangent_slip must be finite")
                if self.contact_partner is None:
                    raise ValueError("contact_tangent_slip requires contact_partner")
                slip_norm = np.linalg.norm(self.contact_tangent_slip, axis=1)
                slip_nodes = slip_norm > 1.0e-12
                if np.any(~np.any(self.contact_candidate[slip_nodes], axis=1)):
                    raise ValueError("contact_tangent_slip nodes must also be contact candidates")
                if np.any(self.contact_partner[slip_nodes] < 0):
                    raise ValueError("contact_tangent_slip nodes must have valid partners")
                if np.any((np.abs(self.contact_tangent_slip) > 1.0e-12) & self.contact_candidate):
                    raise ValueError("contact_tangent_slip components must be separate from normal candidates")
                if self.contact_tangent_stick is not None and np.any(
                    (np.abs(self.contact_tangent_slip) > 1.0e-12) & self.contact_tangent_stick
                ):
                    raise ValueError("contact_tangent_slip and contact_tangent_stick cannot share components")
            if self.contact_tangent_friction is not None:
                if self.contact_friction_mu is None:
                    raise ValueError("contact_tangent_friction requires contact_friction_mu")
                if self.contact_tangent_friction.shape != (n_nodes, 3):
                    raise ValueError("contact_tangent_friction must have shape (N, 3)")
                if self.contact_tangent_friction.dtype != bool:
                    raise ValueError("contact_tangent_friction must be bool array")
                if self.contact_partner is None:
                    raise ValueError("contact_tangent_friction requires contact_partner")
                friction_nodes = np.any(self.contact_tangent_friction, axis=1)
                if np.any(~np.any(self.contact_candidate[friction_nodes], axis=1)):
                    raise ValueError("contact_tangent_friction nodes must also be contact candidates")
                if np.any(self.contact_partner[friction_nodes] < 0):
                    raise ValueError("contact_tangent_friction nodes must have valid partners")
                if np.any(self.contact_tangent_friction & self.contact_candidate):
                    raise ValueError("contact_tangent_friction components must be separate from normal candidates")
                if self.contact_tangent_stick is not None and np.any(
                    self.contact_tangent_friction & self.contact_tangent_stick
                ):
                    raise ValueError("contact_tangent_friction and contact_tangent_stick cannot share components")
                if self.contact_tangent_slip is not None and np.any(
                    self.contact_tangent_friction & (np.abs(self.contact_tangent_slip) > 1.0e-12)
                ):
                    raise ValueError("contact_tangent_friction and contact_tangent_slip cannot share components")

        if self.elements_c3d8.ndim != 2 or self.elements_c3d8.shape[1] != 8:
            raise ValueError("elements_c3d8 must have shape (E8, 8)")
        if self.elements_c3d6.ndim != 2 or self.elements_c3d6.shape[1] != 6:
            raise ValueError("elements_c3d6 must have shape (E6, 6)")

        e8 = int(self.elements_c3d8.shape[0])
        e6 = int(self.elements_c3d6.shape[0])

        if self.E_c3d8.shape != (e8,):
            raise ValueError("E_c3d8 must have shape (E8,)")
        if self.nu_c3d8.shape != (e8,):
            raise ValueError("nu_c3d8 must have shape (E8,)")
        if self.E_c3d6.shape != (e6,):
            raise ValueError("E_c3d6 must have shape (E6,)")
        if self.nu_c3d6.shape != (e6,):
            raise ValueError("nu_c3d6 must have shape (E6,)")

        if np.any(self.E_c3d8 <= 0.0) or np.any(self.E_c3d6 <= 0.0):
            raise ValueError("Young's modulus must be > 0")

        if np.any(self.nu_c3d8 <= -0.99) or np.any(self.nu_c3d8 >= 0.49):
            raise ValueError("nu_c3d8 values must be in (-0.99, 0.49)")
        if np.any(self.nu_c3d6 <= -0.99) or np.any(self.nu_c3d6 >= 0.49):
            raise ValueError("nu_c3d6 values must be in (-0.99, 0.49)")

        for name, elems, width in (
            ("C3D8", self.elements_c3d8, 8),
            ("C3D6", self.elements_c3d6, 6),
        ):
            if elems.shape[1] != width:
                raise ValueError(f"{name} elements must have width {width}")
            if elems.size == 0:
                continue
            if np.any(elems < 0) or np.any(elems >= n_nodes):
                raise ValueError(f"{name} elements contain out-of-range node indices")
            repeated = np.any(np.sort(elems, axis=1)[:, 1:] == np.sort(elems, axis=1)[:, :-1], axis=1)
            if np.any(repeated):
                raise ValueError(f"{name} element has repeated node indices")

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_c3d8(self) -> int:
        return int(self.elements_c3d8.shape[0])

    @property
    def n_c3d6(self) -> int:
        return int(self.elements_c3d6.shape[0])

    @property
    def n_elements(self) -> int:
        return self.n_c3d8 + self.n_c3d6
