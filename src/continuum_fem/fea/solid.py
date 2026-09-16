from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from ..models.solid import Solid


@dataclass(slots=True)
class SolidFEAResult:
    u: np.ndarray
    reactions: np.ndarray
    elem_vm_stress: np.ndarray
    elem_strain_energy: np.ndarray
    gauss_strain: np.ndarray | None = None
    gauss_stress: np.ndarray | None = None
    gauss_vm_stress: np.ndarray | None = None
    gauss_strain_energy: np.ndarray | None = None
    gauss_weight_volume: np.ndarray | None = None

    def as_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {
            "u": self.u,
            "reactions": self.reactions,
            "elem_vm_stress": self.elem_vm_stress,
            "elem_strain_energy": self.elem_strain_energy,
        }
        for key in (
            "gauss_strain",
            "gauss_stress",
            "gauss_vm_stress",
            "gauss_strain_energy",
            "gauss_weight_volume",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass(slots=True)
class SolidContactResult:
    """Rigid-plane contact solve result for bounded pilot benchmarks."""

    fea_result: SolidFEAResult
    active_contact: np.ndarray
    iterations: int
    converged: bool
    active_solid: Solid
    system: Dict[str, np.ndarray | csr_matrix]

    @property
    def u(self) -> np.ndarray:
        return self.fea_result.u

    @property
    def reactions(self) -> np.ndarray:
        return self.fea_result.reactions


class LinearSolidSolver:
    """Linear static mixed-solid FE solver for C3D8 + C3D6 (small strain)."""

    DETJ_TOL = 1e-12
    _CONTACT_ROW_NORMAL = 0
    _CONTACT_ROW_TANGENT_STICK = 1
    _CONTACT_ROW_FRICTION_STICK = 2

    _HEX_SIGN = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    _HEX_GAUSS = [
        (-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0),
        (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0),
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0),
        (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0),
        (-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0),
        (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0),
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0),
        (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0),
    ]

    _WEDGE_TRI_GAUSS = [
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ]
    _WEDGE_LINE_GAUSS = [(-1.0 / np.sqrt(3.0), 1.0), (1.0 / np.sqrt(3.0), 1.0)]

    @staticmethod
    def constitutive_matrix(E: float, nu: float) -> np.ndarray:
        if E <= 0.0:
            raise ValueError("E must be > 0")
        if not (-0.99 < nu < 0.49):
            raise ValueError("nu must be in (-0.99, 0.49)")
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        D = np.zeros((6, 6), dtype=np.float64)
        D[0, 0] = lam + 2.0 * mu
        D[1, 1] = lam + 2.0 * mu
        D[2, 2] = lam + 2.0 * mu
        D[0, 1] = lam
        D[0, 2] = lam
        D[1, 0] = lam
        D[1, 2] = lam
        D[2, 0] = lam
        D[2, 1] = lam
        D[3, 3] = mu
        D[4, 4] = mu
        D[5, 5] = mu
        return D

    @classmethod
    def _c3d8_shape_grad_nat(cls, xi: float, eta: float, zeta: float) -> np.ndarray:
        s = cls._HEX_SIGN
        d = np.zeros((8, 3), dtype=np.float64)
        d[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
        d[:, 1] = 0.125 * s[:, 1] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 2] * zeta)
        d[:, 2] = 0.125 * s[:, 2] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta)
        return d

    @staticmethod
    def _c3d6_shape_grad_nat(r: float, s: float, t: float) -> np.ndarray:
        d = np.zeros((6, 3), dtype=np.float64)
        # dN/dr
        d[0, 0] = -0.5 * (1.0 - t)
        d[1, 0] = 0.5 * (1.0 - t)
        d[2, 0] = 0.0
        d[3, 0] = -0.5 * (1.0 + t)
        d[4, 0] = 0.5 * (1.0 + t)
        d[5, 0] = 0.0
        # dN/ds
        d[0, 1] = -0.5 * (1.0 - t)
        d[1, 1] = 0.0
        d[2, 1] = 0.5 * (1.0 - t)
        d[3, 1] = -0.5 * (1.0 + t)
        d[4, 1] = 0.0
        d[5, 1] = 0.5 * (1.0 + t)
        # dN/dt
        d[0, 2] = -0.5 * (1.0 - r - s)
        d[1, 2] = -0.5 * r
        d[2, 2] = -0.5 * s
        d[3, 2] = 0.5 * (1.0 - r - s)
        d[4, 2] = 0.5 * r
        d[5, 2] = 0.5 * s
        return d

    @staticmethod
    def _build_B(dndx: np.ndarray) -> np.ndarray:
        n_nodes = dndx.shape[0]
        B = np.zeros((6, 3 * n_nodes), dtype=np.float64)
        for i in range(n_nodes):
            ix = 3 * i
            dx, dy, dz = dndx[i, 0], dndx[i, 1], dndx[i, 2]
            B[0, ix + 0] = dx
            B[1, ix + 1] = dy
            B[2, ix + 2] = dz
            B[3, ix + 0] = dy
            B[3, ix + 1] = dx
            B[4, ix + 1] = dz
            B[4, ix + 2] = dy
            B[5, ix + 0] = dz
            B[5, ix + 2] = dx
        return B

    @classmethod
    def _element_stiffness_c3d8(cls, x: np.ndarray, E: float, nu: float) -> np.ndarray:
        D = cls.constitutive_matrix(E, nu)
        ke = np.zeros((24, 24), dtype=np.float64)

        for xi, eta, zeta, w in cls._HEX_GAUSS:
            dnat = cls._c3d8_shape_grad_nat(xi, eta, zeta)
            J = x.T @ dnat
            detJ = float(np.linalg.det(J))
            if detJ <= cls.DETJ_TOL:
                raise ValueError("Invalid C3D8 Jacobian (detJ <= tol)")
            dndx = dnat @ np.linalg.inv(J)
            B = cls._build_B(dndx)
            ke += (B.T @ D @ B) * detJ * w

        return ke

    @classmethod
    def _element_stiffness_c3d6(cls, x: np.ndarray, E: float, nu: float) -> np.ndarray:
        D = cls.constitutive_matrix(E, nu)
        ke = np.zeros((18, 18), dtype=np.float64)

        for r, s, wt in cls._WEDGE_TRI_GAUSS:
            for t, wl in cls._WEDGE_LINE_GAUSS:
                dnat = cls._c3d6_shape_grad_nat(r, s, t)
                J = x.T @ dnat
                detJ = float(np.linalg.det(J))
                if detJ <= cls.DETJ_TOL:
                    raise ValueError("Invalid C3D6 Jacobian (detJ <= tol)")
                dndx = dnat @ np.linalg.inv(J)
                B = cls._build_B(dndx)
                ke += (B.T @ D @ B) * detJ * (wt * wl)

        return ke

    @classmethod
    def _element_stiffness(cls, x: np.ndarray, E: float, nu: float, nper: int) -> np.ndarray:
        if nper == 8:
            return cls._element_stiffness_c3d8(x, E, nu)
        if nper == 6:
            return cls._element_stiffness_c3d6(x, E, nu)
        raise ValueError(f"Unsupported element size {nper}")

    @classmethod
    def _element_vm_avg(cls, x: np.ndarray, ue: np.ndarray, E: float, nu: float, nper: int) -> float:
        D = cls.constitutive_matrix(E, nu)
        vm_vals: list[float] = []

        if nper == 8:
            points: Iterable[Tuple[float, float, float, float]] = cls._HEX_GAUSS
            for xi, eta, zeta, _w in points:
                dnat = cls._c3d8_shape_grad_nat(xi, eta, zeta)
                J = x.T @ dnat
                detJ = float(np.linalg.det(J))
                if detJ <= cls.DETJ_TOL:
                    raise ValueError("Invalid C3D8 Jacobian during stress postprocess")
                dndx = dnat @ np.linalg.inv(J)
                B = cls._build_B(dndx)
                stress = D @ (B @ ue)
                vm_vals.append(cls._von_mises(stress))
            return float(np.mean(vm_vals))

        if nper == 6:
            for r, s, _wt in cls._WEDGE_TRI_GAUSS:
                for t, _wl in cls._WEDGE_LINE_GAUSS:
                    dnat = cls._c3d6_shape_grad_nat(r, s, t)
                    J = x.T @ dnat
                    detJ = float(np.linalg.det(J))
                    if detJ <= cls.DETJ_TOL:
                        raise ValueError("Invalid C3D6 Jacobian during stress postprocess")
                    dndx = dnat @ np.linalg.inv(J)
                    B = cls._build_B(dndx)
                    stress = D @ (B @ ue)
                    vm_vals.append(cls._von_mises(stress))
            return float(np.mean(vm_vals))

        raise ValueError(f"Unsupported element size {nper}")

    @classmethod
    def _element_gauss_state(
        cls,
        x: np.ndarray,
        ue: np.ndarray,
        E: float,
        nu: float,
        nper: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return engineering strain/stress and weighted energy at element quadrature points."""
        D = cls.constitutive_matrix(E, nu)
        strain_vals: list[np.ndarray] = []
        stress_vals: list[np.ndarray] = []
        vm_vals: list[float] = []
        energy_vals: list[float] = []
        weight_vol_vals: list[float] = []

        def append_state(dnat: np.ndarray, weight: float, error_context: str) -> None:
            J = x.T @ dnat
            detJ = float(np.linalg.det(J))
            if detJ <= cls.DETJ_TOL:
                raise ValueError(f"Invalid {error_context} Jacobian during FE-state postprocess")
            dndx = dnat @ np.linalg.inv(J)
            B = cls._build_B(dndx)
            strain = B @ ue
            stress = D @ strain
            weight_volume = detJ * float(weight)
            strain_vals.append(np.asarray(strain, dtype=np.float64))
            stress_vals.append(np.asarray(stress, dtype=np.float64))
            vm_vals.append(cls._von_mises(stress))
            energy_vals.append(0.5 * float(strain @ stress) * weight_volume)
            weight_vol_vals.append(weight_volume)

        if nper == 8:
            for xi, eta, zeta, weight in cls._HEX_GAUSS:
                append_state(cls._c3d8_shape_grad_nat(xi, eta, zeta), float(weight), "C3D8")
        elif nper == 6:
            for r, s, wt in cls._WEDGE_TRI_GAUSS:
                for t, wl in cls._WEDGE_LINE_GAUSS:
                    append_state(cls._c3d6_shape_grad_nat(r, s, t), float(wt) * float(wl), "C3D6")
        else:
            raise ValueError(f"Unsupported element size {nper}")

        return (
            np.asarray(strain_vals, dtype=np.float64),
            np.asarray(stress_vals, dtype=np.float64),
            np.asarray(vm_vals, dtype=np.float64).reshape(-1, 1),
            np.asarray(energy_vals, dtype=np.float64).reshape(-1, 1),
            np.asarray(weight_vol_vals, dtype=np.float64).reshape(-1, 1),
        )

    @staticmethod
    def _von_mises(stress: np.ndarray) -> float:
        sxx, syy, szz, sxy, syz, szx = [float(v) for v in stress]
        return float(
            np.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy**2 + syz**2 + szx**2)
            )
        )

    @classmethod
    def assemble_global_K(cls, solid: Solid) -> csr_matrix:
        solid.validate()

        n_dofs = 3 * solid.n_nodes
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def push_element(elem_nodes: np.ndarray, E: float, nu: float) -> None:
            x = solid.nodes[elem_nodes]
            nper = int(elem_nodes.shape[0])
            ke = cls._element_stiffness(x, E=E, nu=nu, nper=nper)
            dofs = np.empty((3 * nper,), dtype=np.int64)
            for i, node_id in enumerate(elem_nodes):
                base = 3 * int(node_id)
                dofs[3 * i + 0] = base + 0
                dofs[3 * i + 1] = base + 1
                dofs[3 * i + 2] = base + 2
            rr = np.repeat(dofs, dofs.size)
            cc = np.tile(dofs, dofs.size)
            rows.append(rr)
            cols.append(cc)
            vals.append(ke.reshape(-1))

        for eidx, elem in enumerate(solid.elements_c3d8):
            push_element(elem, E=float(solid.E_c3d8[eidx]), nu=float(solid.nu_c3d8[eidx]))

        for eidx, elem in enumerate(solid.elements_c3d6):
            push_element(elem, E=float(solid.E_c3d6[eidx]), nu=float(solid.nu_c3d6[eidx]))

        if not rows:
            raise ValueError("Solid has no elements to assemble")

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(vals)
        K = coo_matrix((data, (row, col)), shape=(n_dofs, n_dofs), dtype=np.float64).tocsr()
        return K

    @classmethod
    def build_system(cls, solid: Solid) -> Dict[str, np.ndarray | csr_matrix]:
        K = cls.assemble_global_K(solid)
        F = solid.loads.reshape(-1).astype(np.float64)
        fixed_mask = solid.fixed.reshape(-1)
        u_prescribed = np.zeros((3 * solid.n_nodes,), dtype=np.float64)
        if solid.prescribed_displacements is not None:
            u_prescribed[:] = np.asarray(solid.prescribed_displacements, dtype=np.float64).reshape(-1)
        free = np.where(~fixed_mask)[0]
        fixed = np.where(fixed_mask)[0]
        if free.size == 0:
            raise ValueError("All DOFs are fixed")

        Kff = K[free][:, free].tocsr()
        Ff = F[free].copy()
        if fixed.size > 0 and bool(np.any(np.abs(u_prescribed[fixed]) > 0.0)):
            Ff = Ff - np.asarray(K[free][:, fixed] @ u_prescribed[fixed], dtype=np.float64)
        return {
            "K": K,
            "F": F,
            "free": free,
            "fixed": fixed,
            "fixed_mask": fixed_mask,
            "u_prescribed": u_prescribed,
            "Kff": Kff,
            "Ff": Ff,
        }

    @staticmethod
    def _with_boundary_conditions(
        solid: Solid,
        *,
        fixed: np.ndarray,
        prescribed_displacements: np.ndarray,
    ) -> Solid:
        return Solid(
            nodes=solid.nodes,
            elements_c3d8=solid.elements_c3d8,
            elements_c3d6=solid.elements_c3d6,
            E_c3d8=solid.E_c3d8,
            nu_c3d8=solid.nu_c3d8,
            E_c3d6=solid.E_c3d6,
            nu_c3d6=solid.nu_c3d6,
            loads=solid.loads,
            fixed=np.asarray(fixed, dtype=bool),
            prescribed_displacements=np.asarray(prescribed_displacements, dtype=np.float64),
            contact_candidate=solid.contact_candidate,
            contact_normal=solid.contact_normal,
            contact_gap=solid.contact_gap,
            contact_partner=solid.contact_partner,
            contact_tangent_stick=solid.contact_tangent_stick,
            contact_tangent_slip=solid.contact_tangent_slip,
            contact_tangent_friction=solid.contact_tangent_friction,
            contact_friction_mu=solid.contact_friction_mu,
            contact_incremental_steps=solid.contact_incremental_steps,
            contact_partner_remap=solid.contact_partner_remap,
            contact_dynamic_steps=solid.contact_dynamic_steps,
            contact_dynamic_dt=solid.contact_dynamic_dt,
            contact_dynamic_damping=solid.contact_dynamic_damping,
            contact_dynamic_mass=solid.contact_dynamic_mass,
            contact_penalty_stiffness=solid.contact_penalty_stiffness,
            contact_penalty_exponent=solid.contact_penalty_exponent,
        )

    @staticmethod
    def _scaled_contact_step_solid(
        solid: Solid,
        alpha: float,
        *,
        contact_partner: np.ndarray | None = None,
    ) -> Solid:
        prescribed = (
            np.zeros((solid.n_nodes, 3), dtype=np.float64)
            if solid.prescribed_displacements is None
            else np.asarray(solid.prescribed_displacements, dtype=np.float64)
        )
        return Solid(
            nodes=solid.nodes,
            elements_c3d8=solid.elements_c3d8,
            elements_c3d6=solid.elements_c3d6,
            E_c3d8=solid.E_c3d8,
            nu_c3d8=solid.nu_c3d8,
            E_c3d6=solid.E_c3d6,
            nu_c3d6=solid.nu_c3d6,
            loads=np.asarray(solid.loads, dtype=np.float64) * float(alpha),
            fixed=np.asarray(solid.fixed, dtype=bool),
            prescribed_displacements=prescribed * float(alpha),
            contact_candidate=solid.contact_candidate,
            contact_normal=solid.contact_normal,
            contact_gap=solid.contact_gap,
            contact_partner=(
                solid.contact_partner
                if contact_partner is None
                else np.asarray(contact_partner, dtype=np.int64).reshape(solid.n_nodes)
            ),
            contact_tangent_stick=solid.contact_tangent_stick,
            contact_tangent_slip=solid.contact_tangent_slip,
            contact_tangent_friction=solid.contact_tangent_friction,
            contact_friction_mu=solid.contact_friction_mu,
            contact_incremental_steps=solid.contact_incremental_steps,
            contact_partner_remap=solid.contact_partner_remap,
            contact_dynamic_steps=solid.contact_dynamic_steps,
            contact_dynamic_dt=solid.contact_dynamic_dt,
            contact_dynamic_damping=solid.contact_dynamic_damping,
            contact_dynamic_mass=solid.contact_dynamic_mass,
            contact_penalty_stiffness=solid.contact_penalty_stiffness,
            contact_penalty_exponent=solid.contact_penalty_exponent,
        )

    @staticmethod
    def _remapped_contact_partner(solid: Solid, u: np.ndarray | None) -> np.ndarray:
        partner = (
            np.asarray(solid.contact_partner, dtype=np.int64).reshape(solid.n_nodes).copy()
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        if not bool(solid.contact_partner_remap):
            return partner
        if solid.contact_candidate is None:
            return partner
        candidate_nodes = np.where(np.any(np.asarray(solid.contact_candidate, dtype=bool), axis=1))[0]
        if candidate_nodes.size == 0:
            return partner
        master_pool = np.unique(partner[candidate_nodes][partner[candidate_nodes] >= 0]).astype(np.int64, copy=False)
        if master_pool.size == 0:
            return partner

        reference = np.asarray(solid.nodes, dtype=np.float64).reshape(solid.n_nodes, 3)
        if u is None:
            deformed = reference
        else:
            deformed = reference + np.asarray(u, dtype=np.float64).reshape(solid.n_nodes, 3)
        z_span = float(np.max(reference[:, 2]) - np.min(reference[:, 2])) if reference.size else 0.0
        z_tol = max(1.0e-9, 1.0e-8 * max(1.0, abs(z_span)))
        for node_id in candidate_nodes:
            old_partner = int(partner[int(node_id)])
            if old_partner >= 0:
                z_ref = float(reference[old_partner, 2])
            else:
                z_ref = float(reference[int(node_id), 2])
            same_z = master_pool[np.abs(reference[master_pool, 2] - z_ref) <= z_tol]
            pool = same_z if same_z.size > 0 else master_pool
            delta = deformed[pool][:, [0, 2]] - deformed[int(node_id), [0, 2]].reshape(1, 2)
            partner[int(node_id)] = int(pool[int(np.argmin(np.sum(delta * delta, axis=1)))])
        return partner

    @staticmethod
    def _contact_constraint_rows(solid: Solid) -> list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]]:
        if solid.contact_candidate is None or solid.contact_normal is None or solid.contact_gap is None:
            raise ValueError("solid has no contact metadata")
        candidate = np.asarray(solid.contact_candidate, dtype=bool) & ~np.asarray(solid.fixed, dtype=bool)
        normal = np.asarray(solid.contact_normal, dtype=np.float64)
        gap = np.asarray(solid.contact_gap, dtype=np.float64)
        partner = (
            np.asarray(solid.contact_partner, dtype=np.int64)
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]] = []
        for node_id in range(int(solid.n_nodes)):
            comps = np.flatnonzero(candidate[node_id])
            if comps.size == 0:
                continue
            coeffs = normal[node_id, comps].astype(np.float64, copy=True)
            norm = float(np.linalg.norm(coeffs))
            if norm <= 1.0e-30:
                continue
            dofs = 3 * int(node_id) + comps.astype(np.int64, copy=True)
            row_coeffs = coeffs.copy()
            partner_id = int(partner[node_id])
            if partner_id >= 0:
                dofs = np.concatenate([dofs, 3 * partner_id + comps.astype(np.int64, copy=True)])
                row_coeffs = np.concatenate([row_coeffs, -coeffs])
            gap_value = float(np.mean(gap[node_id, comps]))
            constraints.append((dofs, row_coeffs, gap_value, int(node_id), comps.astype(np.int64, copy=True)))
        return constraints

    @staticmethod
    def _contact_equality_rows(
        solid: Solid,
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]],
        active_constraints: np.ndarray,
        friction_slip_state: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, tuple[int, int, int, int]]]:
        rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, tuple[int, int, int, int]]] = []
        stick = (
            np.asarray(solid.contact_tangent_stick, dtype=bool)
            if solid.contact_tangent_stick is not None
            else np.zeros((solid.n_nodes, 3), dtype=bool)
        )
        slip = (
            np.asarray(solid.contact_tangent_slip, dtype=np.float64)
            if solid.contact_tangent_slip is not None
            else np.zeros((solid.n_nodes, 3), dtype=np.float64)
        )
        friction = (
            np.asarray(solid.contact_tangent_friction, dtype=bool)
            if solid.contact_tangent_friction is not None
            else np.zeros((solid.n_nodes, 3), dtype=bool)
        )
        friction_mu = (
            np.asarray(solid.contact_friction_mu, dtype=np.float64)
            if solid.contact_friction_mu is not None
            else np.zeros((solid.n_nodes,), dtype=np.float64)
        )
        partner = (
            np.asarray(solid.contact_partner, dtype=np.int64)
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        slip_state = friction_slip_state or {}
        for constraint_idx in np.flatnonzero(active_constraints):
            dofs, coeffs, gap_value, node_id, _comps = constraints[int(constraint_idx)]
            node_id = int(node_id)
            partner_id = int(partner[int(node_id)])
            force_dofs = dofs.copy()
            force_coeffs = coeffs.copy()
            slip_norm = float(np.linalg.norm(slip[int(node_id)]))
            if partner_id >= 0 and slip_norm > 1.0e-12 and float(friction_mu[int(node_id)]) > 0.0:
                tangent = slip[int(node_id)] / slip_norm
                tangent_comps = np.flatnonzero(np.abs(tangent) > 1.0e-12)
                tangent_dofs = np.concatenate(
                    [
                        3 * int(node_id) + tangent_comps.astype(np.int64, copy=True),
                        3 * partner_id + tangent_comps.astype(np.int64, copy=True),
                    ]
                )
                tangent_coeffs = np.concatenate([tangent[tangent_comps], -tangent[tangent_comps]])
                force_dofs = np.concatenate([force_dofs, tangent_dofs])
                force_coeffs = np.concatenate(
                    [force_coeffs, -float(friction_mu[int(node_id)]) * tangent_coeffs.astype(np.float64)]
                )
            if partner_id >= 0 and float(friction_mu[node_id]) > 0.0:
                for comp in np.flatnonzero(friction[node_id]):
                    direction = float(slip_state.get((node_id, int(comp)), 0.0))
                    if abs(direction) <= 1.0e-12:
                        continue
                    tangent_dofs = np.asarray([3 * node_id + int(comp), 3 * partner_id + int(comp)], dtype=np.int64)
                    tangent_coeffs = np.asarray([direction, -direction], dtype=np.float64)
                    force_dofs = np.concatenate([force_dofs, tangent_dofs])
                    force_coeffs = np.concatenate(
                        [force_coeffs, -float(friction_mu[node_id]) * tangent_coeffs.astype(np.float64)]
                    )
            rows.append(
                (
                    dofs,
                    coeffs,
                    force_dofs,
                    force_coeffs,
                    -float(gap_value),
                    (LinearSolidSolver._CONTACT_ROW_NORMAL, node_id, -1, int(constraint_idx)),
                )
            )
            if partner_id < 0:
                continue
            for comp in np.flatnonzero(stick[int(node_id)]):
                rows.append(
                    (
                        np.asarray([3 * int(node_id) + int(comp), 3 * partner_id + int(comp)], dtype=np.int64),
                        np.asarray([1.0, -1.0], dtype=np.float64),
                        np.asarray([3 * int(node_id) + int(comp), 3 * partner_id + int(comp)], dtype=np.int64),
                        np.asarray([1.0, -1.0], dtype=np.float64),
                        0.0,
                        (
                            LinearSolidSolver._CONTACT_ROW_TANGENT_STICK,
                            int(node_id),
                            int(comp),
                            int(constraint_idx),
                        ),
                    )
                )
            for comp in np.flatnonzero(friction[node_id]):
                if abs(float(slip_state.get((node_id, int(comp)), 0.0))) > 1.0e-12:
                    continue
                rows.append(
                    (
                        np.asarray([3 * node_id + int(comp), 3 * partner_id + int(comp)], dtype=np.int64),
                        np.asarray([1.0, -1.0], dtype=np.float64),
                        np.asarray([3 * node_id + int(comp), 3 * partner_id + int(comp)], dtype=np.int64),
                        np.asarray([1.0, -1.0], dtype=np.float64),
                        0.0,
                        (
                            LinearSolidSolver._CONTACT_ROW_FRICTION_STICK,
                            node_id,
                            int(comp),
                            int(constraint_idx),
                        ),
                    )
                )
        return rows

    @staticmethod
    def _contact_signed_gaps(
        u: np.ndarray,
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]],
    ) -> np.ndarray:
        u_flat = np.asarray(u, dtype=np.float64).reshape(-1)
        values = np.empty((len(constraints),), dtype=np.float64)
        for idx, (dofs, coeffs, gap_value, _node_id, _comps) in enumerate(constraints):
            values[idx] = float(gap_value + coeffs @ u_flat[dofs])
        return values

    @staticmethod
    def _active_contact_mask(
        solid: Solid,
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]],
        active_constraints: np.ndarray,
    ) -> np.ndarray:
        active = np.zeros((solid.n_nodes, 3), dtype=bool)
        for idx, (_dofs, _coeffs, _gap_value, node_id, comps) in enumerate(constraints):
            if bool(active_constraints[idx]):
                active[int(node_id), comps] = True
        return active

    @staticmethod
    def _contact_friction_arrays(solid: Solid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        friction = (
            np.asarray(solid.contact_tangent_friction, dtype=bool)
            if solid.contact_tangent_friction is not None
            else np.zeros((solid.n_nodes, 3), dtype=bool)
        )
        friction_mu = (
            np.asarray(solid.contact_friction_mu, dtype=np.float64)
            if solid.contact_friction_mu is not None
            else np.zeros((solid.n_nodes,), dtype=np.float64)
        )
        partner = (
            np.asarray(solid.contact_partner, dtype=np.int64)
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        return friction, friction_mu, partner

    @staticmethod
    def _contact_friction_direction(solid: Solid, u: np.ndarray, node_id: int, partner_id: int, comp: int) -> float:
        rel = float(u[int(node_id), int(comp)] - u[int(partner_id), int(comp)])
        if abs(rel) <= 1.0e-12 and solid.prescribed_displacements is not None:
            prescribed = np.asarray(solid.prescribed_displacements, dtype=np.float64).reshape(solid.n_nodes, 3)
            rel = float(prescribed[int(node_id), int(comp)] - prescribed[int(partner_id), int(comp)])
        if abs(rel) <= 1.0e-12:
            return 1.0
        return 1.0 if rel > 0.0 else -1.0

    @classmethod
    def _initialize_contact_friction_state(
        cls,
        solid: Solid,
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]],
        newly_active: np.ndarray,
        u_trial: np.ndarray,
        friction_slip_state: dict[tuple[int, int], float],
        friction_trial_direction: dict[tuple[int, int], float],
    ) -> None:
        friction, friction_mu, partner = cls._contact_friction_arrays(solid)
        if not bool(np.any(friction)):
            return
        for constraint_idx in np.flatnonzero(newly_active):
            _dofs, _coeffs, _gap_value, node_id, _comps = constraints[int(constraint_idx)]
            node_id = int(node_id)
            partner_id = int(partner[node_id])
            if partner_id < 0 or float(friction_mu[node_id]) < 0.0:
                continue
            for comp in np.flatnonzero(friction[node_id]):
                key = (node_id, int(comp))
                friction_slip_state.setdefault(key, 0.0)
                friction_trial_direction[key] = cls._contact_friction_direction(
                    solid,
                    u_trial,
                    node_id,
                    partner_id,
                    int(comp),
                )

    @classmethod
    def _contact_friction_slip_updates(
        cls,
        solid: Solid,
        system: Dict[str, np.ndarray | csr_matrix],
        friction_slip_state: dict[tuple[int, int], float],
        friction_trial_direction: dict[tuple[int, int], float],
    ) -> bool:
        friction, friction_mu, _partner = cls._contact_friction_arrays(solid)
        if not bool(np.any(friction)):
            return False
        kind = np.asarray(system.get("contact_row_kind", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        node = np.asarray(system.get("contact_row_node", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        comp = np.asarray(system.get("contact_row_component", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        constraint = np.asarray(system.get("contact_row_constraint", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        multipliers = np.asarray(system.get("contact_multipliers", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        normal_force: dict[int, float] = {}
        for idx in np.flatnonzero(kind == cls._CONTACT_ROW_NORMAL):
            normal_force[int(constraint[int(idx)])] = abs(float(multipliers[int(idx)]))
        changed = False
        for idx in np.flatnonzero(kind == cls._CONTACT_ROW_FRICTION_STICK):
            node_id = int(node[int(idx)])
            comp_id = int(comp[int(idx)])
            key = (node_id, comp_id)
            if abs(float(friction_slip_state.get(key, 0.0))) > 1.0e-12:
                continue
            normal = normal_force.get(int(constraint[int(idx)]), 0.0)
            limit = float(friction_mu[node_id]) * float(normal)
            tangent = abs(float(multipliers[int(idx)]))
            tol = 1.0e-8 * max(1.0, float(limit), float(tangent), float(normal))
            if tangent <= limit + tol:
                continue
            direction = float(friction_trial_direction.get(key, 0.0))
            if abs(direction) <= 1.0e-12:
                direction = -1.0 if float(multipliers[int(idx)]) > 0.0 else 1.0
            friction_slip_state[key] = 1.0 if direction > 0.0 else -1.0
            changed = True
        return changed

    @classmethod
    def _attach_contact_friction_state(
        cls,
        solid: Solid,
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]],
        active_constraints: np.ndarray,
        friction_slip_state: dict[tuple[int, int], float],
        system: Dict[str, np.ndarray | csr_matrix],
    ) -> None:
        friction, _friction_mu, _partner = cls._contact_friction_arrays(solid)
        slip_direction = np.zeros((solid.n_nodes, 3), dtype=np.float64)
        stick = np.zeros((solid.n_nodes, 3), dtype=bool)
        for constraint_idx in np.flatnonzero(active_constraints):
            _dofs, _coeffs, _gap_value, node_id, _comps = constraints[int(constraint_idx)]
            node_id = int(node_id)
            for comp in np.flatnonzero(friction[node_id]):
                direction = float(friction_slip_state.get((node_id, int(comp)), 0.0))
                if abs(direction) > 1.0e-12:
                    slip_direction[node_id, int(comp)] = 1.0 if direction > 0.0 else -1.0
                else:
                    stick[node_id, int(comp)] = True
        system["contact_friction_slip_direction"] = slip_direction
        system["contact_friction_stick"] = stick

    @classmethod
    def _solve_with_contact_constraints(
        cls,
        solid: Solid,
        constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]],
        active_constraints: np.ndarray,
        friction_slip_state: dict[tuple[int, int], float] | None = None,
        *,
        system_override: Dict[str, np.ndarray | csr_matrix] | None = None,
        kff_override: csr_matrix | None = None,
        ff_override: np.ndarray | None = None,
    ) -> tuple[SolidFEAResult, Dict[str, np.ndarray | csr_matrix]]:
        system = cls.build_system(solid) if system_override is None else dict(system_override)
        K = system["K"]
        F = system["F"]
        Kff: csr_matrix = system["Kff"] if kff_override is None else kff_override
        Ff = np.asarray(system["Ff"] if ff_override is None else ff_override, dtype=np.float64)
        free = np.asarray(system["free"], dtype=np.int64)
        u_prescribed = np.asarray(system["u_prescribed"], dtype=np.float64).copy()
        if Kff.shape != (int(free.size), int(free.size)):
            raise ValueError("contact Kff override shape does not match free DOF count")
        if Ff.shape != (int(free.size),):
            raise ValueError("contact Ff override shape does not match free DOF count")
        full_to_free = np.full((3 * solid.n_nodes,), -1, dtype=np.int64)
        full_to_free[free] = np.arange(int(free.size), dtype=np.int64)

        equality_rows = cls._contact_equality_rows(
            solid,
            constraints,
            active_constraints,
            friction_slip_state=friction_slip_state,
        )
        row_meta: list[tuple[int, int, int, int]] = []
        if equality_rows:
            row_idx: list[int] = []
            col_idx: list[int] = []
            values: list[float] = []
            force_row_idx: list[int] = []
            force_col_idx: list[int] = []
            force_values: list[float] = []
            rhs_values: list[float] = []
            for dofs, coeffs, force_dofs, force_coeffs, rhs_target, meta in equality_rows:
                row = len(rhs_values)
                rhs_value = float(rhs_target)
                has_free = False
                for dof, coeff in zip(dofs, coeffs):
                    free_idx = int(full_to_free[dof])
                    if free_idx >= 0:
                        has_free = True
                        row_idx.append(row)
                        col_idx.append(free_idx)
                        values.append(float(coeff))
                    else:
                        rhs_value -= float(coeff) * float(u_prescribed[dof])
                if has_free:
                    for dof, coeff in zip(force_dofs, force_coeffs):
                        free_idx = int(full_to_free[dof])
                        if free_idx >= 0:
                            force_row_idx.append(row)
                            force_col_idx.append(free_idx)
                            force_values.append(float(coeff))
                    rhs_values.append(rhs_value)
                    row_meta.append(meta)
                elif abs(rhs_value) > 1.0e-10:
                    raise ValueError("Contact equality row has no free DOFs and inconsistent prescribed values")
            n_rows = len(rhs_values)
            C = coo_matrix(
                (np.asarray(values, dtype=np.float64), (np.asarray(row_idx), np.asarray(col_idx))),
                shape=(n_rows, int(free.size)),
                dtype=np.float64,
            ).tocsr()
            G = coo_matrix(
                (
                    np.asarray(force_values, dtype=np.float64),
                    (np.asarray(force_row_idx), np.asarray(force_col_idx)),
                ),
                shape=(n_rows, int(free.size)),
                dtype=np.float64,
            ).tocsr()
            zeros = csr_matrix((n_rows, n_rows), dtype=np.float64)
            saddle = bmat([[Kff, G.T], [C, zeros]], format="csr")
            rhs = np.concatenate([Ff, np.asarray(rhs_values, dtype=np.float64)])
            solved = np.asarray(spsolve(saddle, rhs), dtype=np.float64)
            uf = solved[: int(free.size)]
            multipliers = solved[int(free.size) :]
        else:
            uf = np.asarray(spsolve(Kff, Ff), dtype=np.float64)
            multipliers = np.zeros((0,), dtype=np.float64)

        if np.any(~np.isfinite(uf)):
            raise ValueError("Contact solve failed (non-finite displacement result)")
        system = dict(system)
        if row_meta:
            meta_arr = np.asarray(row_meta, dtype=np.int64)
            system["contact_row_kind"] = meta_arr[:, 0]
            system["contact_row_node"] = meta_arr[:, 1]
            system["contact_row_component"] = meta_arr[:, 2]
            system["contact_row_constraint"] = meta_arr[:, 3]
        else:
            system["contact_row_kind"] = np.zeros((0,), dtype=np.int64)
            system["contact_row_node"] = np.zeros((0,), dtype=np.int64)
            system["contact_row_component"] = np.zeros((0,), dtype=np.int64)
            system["contact_row_constraint"] = np.zeros((0,), dtype=np.int64)
        system["contact_multipliers"] = np.asarray(multipliers, dtype=np.float64)
        u_flat = u_prescribed.copy()
        u_flat[free] = uf
        reactions = (K @ u_flat - F).reshape(solid.n_nodes, 3)
        u = u_flat.reshape(solid.n_nodes, 3)
        return cls.postprocess_fields(solid, u, reactions=reactions), system

    @classmethod
    def solve_contact(
        cls,
        solid: Solid,
        *,
        max_active_set_iter: int = 20,
        violation_tol: float = 1.0e-10,
    ) -> SolidContactResult:
        """
        Solve a small-strain unilateral contact pilot problem.

        A row with one contact component is treated as the original axis DOF
        constraint. A row with multiple components is treated as one coupled
        constraint ``gap + normal . u_node >= 0``. If ``contact_partner`` is
        set for a candidate node, the row becomes
        ``gap + normal . (u_node - u_partner) >= 0``. If
        ``contact_tangent_stick`` is set, active paired contacts also enforce
        zero relative displacement on the marked tangential components. If
        ``contact_tangent_slip`` and ``contact_friction_mu`` are set, active
        paired contacts apply a prescribed-direction Coulomb-limit tangential
        force column coupled to the normal multiplier. If
        ``contact_tangent_friction`` and ``contact_friction_mu`` are set,
        active paired contacts start as sticking constraints and are released
        to Coulomb slip when their sticking multiplier exceeds the friction
        limit. The active set is monotone-additive and intended for bounded
        synthetic pilots, not a general frictional contact algorithm.
        """
        solid.validate()
        if max_active_set_iter < 0:
            raise ValueError("max_active_set_iter must be >= 0")
        constraints = cls._contact_constraint_rows(solid)
        if not constraints:
            raise ValueError("contact solve requires at least one non-fixed contact candidate")

        active_constraints = np.zeros((len(constraints),), dtype=bool)
        friction_slip_state: dict[tuple[int, int], float] = {}
        friction_trial_direction: dict[tuple[int, int], float] = {}
        result, system = cls._solve_with_contact_constraints(
            solid,
            constraints,
            active_constraints,
            friction_slip_state=friction_slip_state,
        )
        for iteration in range(int(max_active_set_iter) + 1):
            signed_gap = cls._contact_signed_gaps(result.u, constraints)
            violated = signed_gap < -float(violation_tol)
            new_active = active_constraints | violated
            newly_active = new_active & ~active_constraints
            if not np.array_equal(new_active, active_constraints):
                cls._initialize_contact_friction_state(
                    solid,
                    constraints,
                    newly_active,
                    result.u,
                    friction_slip_state,
                    friction_trial_direction,
                )
                active_constraints = new_active
                result, system = cls._solve_with_contact_constraints(
                    solid,
                    constraints,
                    active_constraints,
                    friction_slip_state=friction_slip_state,
                )
                continue
            if cls._contact_friction_slip_updates(
                solid,
                system,
                friction_slip_state,
                friction_trial_direction,
            ):
                result, system = cls._solve_with_contact_constraints(
                    solid,
                    constraints,
                    active_constraints,
                    friction_slip_state=friction_slip_state,
                )
                continue

            cls._attach_contact_friction_state(
                solid,
                constraints,
                active_constraints,
                friction_slip_state,
                system,
            )
            return SolidContactResult(
                fea_result=result,
                active_contact=cls._active_contact_mask(solid, constraints, active_constraints),
                iterations=int(iteration),
                converged=True,
                active_solid=solid,
                system=system,
            )

        cls._attach_contact_friction_state(
            solid,
            constraints,
            active_constraints,
            friction_slip_state,
            system,
        )
        return SolidContactResult(
            fea_result=result,
            active_contact=cls._active_contact_mask(solid, constraints, active_constraints),
            iterations=int(max_active_set_iter),
            converged=False,
            active_solid=solid,
            system=system,
        )

    @classmethod
    def solve_contact_incremental(
        cls,
        solid: Solid,
        *,
        n_steps: int | None = None,
        max_active_set_iter: int = 20,
        violation_tol: float = 1.0e-10,
    ) -> SolidContactResult:
        """
        Solve contact over a proportional load/prescribed-displacement ramp.

        Normal active constraints and inferred-friction slip releases persist
        from earlier increments. If ``contact_partner_remap`` is enabled, paired
        contact rows are rebuilt from a nearest-master remap based on the
        previous increment's deformed tangential position. This is a small-strain
        synthetic path pilot; it is not a full finite-deformation or general
        return-mapping contact algorithm.
        """
        solid.validate()
        steps = int(n_steps if n_steps is not None else (solid.contact_incremental_steps or 1))
        if steps <= 1:
            return cls.solve_contact(
                solid,
                max_active_set_iter=max_active_set_iter,
                violation_tol=violation_tol,
            )
        if max_active_set_iter < 0:
            raise ValueError("max_active_set_iter must be >= 0")

        active_constraints: np.ndarray | None = None
        friction_slip_state: dict[tuple[int, int], float] = {}
        friction_trial_direction: dict[tuple[int, int], float] = {}
        total_iterations = 0
        result: SolidFEAResult | None = None
        system: Dict[str, np.ndarray | csr_matrix] | None = None
        final_constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]] | None = None
        final_solid = solid

        for step_idx in range(1, steps + 1):
            alpha = float(step_idx) / float(steps)
            remapped_partner = cls._remapped_contact_partner(solid, result.u if result is not None else None)
            step_solid = cls._scaled_contact_step_solid(solid, alpha, contact_partner=remapped_partner)
            constraints = cls._contact_constraint_rows(step_solid)
            if not constraints:
                raise ValueError("incremental contact solve requires at least one non-fixed contact candidate")
            if active_constraints is None:
                active_constraints = np.zeros((len(constraints),), dtype=bool)
            elif len(constraints) != int(active_constraints.size):
                raise ValueError("contact partner remap changed the active-set row count")
            final_constraints = constraints
            final_solid = step_solid
            result, system = cls._solve_with_contact_constraints(
                step_solid,
                constraints,
                active_constraints,
                friction_slip_state=friction_slip_state,
            )
            step_converged = False
            for _iteration in range(int(max_active_set_iter) + 1):
                total_iterations += 1
                signed_gap = cls._contact_signed_gaps(result.u, constraints)
                violated = signed_gap < -float(violation_tol)
                new_active = active_constraints | violated
                newly_active = new_active & ~active_constraints
                if not np.array_equal(new_active, active_constraints):
                    cls._initialize_contact_friction_state(
                        step_solid,
                        constraints,
                        newly_active,
                        result.u,
                        friction_slip_state,
                        friction_trial_direction,
                    )
                    active_constraints = new_active
                    result, system = cls._solve_with_contact_constraints(
                        step_solid,
                        constraints,
                        active_constraints,
                        friction_slip_state=friction_slip_state,
                    )
                    continue
                if cls._contact_friction_slip_updates(
                    step_solid,
                    system,
                    friction_slip_state,
                    friction_trial_direction,
                ):
                    result, system = cls._solve_with_contact_constraints(
                        step_solid,
                        constraints,
                        active_constraints,
                        friction_slip_state=friction_slip_state,
                    )
                    continue
                step_converged = True
                break
            if not step_converged:
                cls._attach_contact_friction_state(
                    step_solid,
                    constraints,
                    active_constraints,
                    friction_slip_state,
                    system,
                )
                system["contact_incremental_steps"] = np.asarray([steps], dtype=np.int64)
                system["contact_partner_initial"] = (
                    np.asarray(solid.contact_partner, dtype=np.int64).copy()
                    if solid.contact_partner is not None
                    else np.full((solid.n_nodes,), -1, dtype=np.int64)
                )
                system["contact_partner_final"] = (
                    np.asarray(step_solid.contact_partner, dtype=np.int64).copy()
                    if step_solid.contact_partner is not None
                    else np.full((step_solid.n_nodes,), -1, dtype=np.int64)
                )
                return SolidContactResult(
                    fea_result=result,
                    active_contact=cls._active_contact_mask(step_solid, constraints, active_constraints),
                    iterations=int(total_iterations),
                    converged=False,
                    active_solid=step_solid,
                    system=system,
                )

        if result is None or system is None or active_constraints is None or final_constraints is None:
            raise ValueError("incremental contact solve did not run any steps")
        cls._attach_contact_friction_state(
            final_solid,
            final_constraints,
            active_constraints,
            friction_slip_state,
            system,
        )
        system["contact_incremental_steps"] = np.asarray([steps], dtype=np.int64)
        system["contact_partner_initial"] = (
            np.asarray(solid.contact_partner, dtype=np.int64).copy()
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        system["contact_partner_final"] = (
            np.asarray(final_solid.contact_partner, dtype=np.int64).copy()
            if final_solid.contact_partner is not None
            else np.full((final_solid.n_nodes,), -1, dtype=np.int64)
        )
        return SolidContactResult(
            fea_result=result,
            active_contact=cls._active_contact_mask(final_solid, final_constraints, active_constraints),
            iterations=int(total_iterations),
            converged=True,
            active_solid=final_solid,
            system=system,
        )

    @classmethod
    def solve_contact_dynamic_relaxation(
        cls,
        solid: Solid,
        *,
        n_steps: int | None = None,
        dt: float | None = None,
        damping: float | None = None,
        max_active_set_iter: int = 20,
        violation_tol: float = 1.0e-10,
    ) -> SolidContactResult:
        """
        Solve a bounded implicit dynamic contact rollout.

        This uses the same monotone active-set/friction rows as
        ``solve_contact_incremental`` but each proportional load step solves
        ``(K + M / dt^2 + C / dt) u`` against the static step load plus the
        previous two displacement states. It is a synthetic small-strain
        dynamic-relaxation pilot, not a full transient nonlinear contact code.
        """
        solid.validate()
        steps = int(n_steps if n_steps is not None else (solid.contact_dynamic_steps or 1))
        if steps <= 1:
            return cls.solve_contact(
                solid,
                max_active_set_iter=max_active_set_iter,
                violation_tol=violation_tol,
            )
        if max_active_set_iter < 0:
            raise ValueError("max_active_set_iter must be >= 0")
        if solid.contact_dynamic_mass is None:
            raise ValueError("dynamic contact solve requires contact_dynamic_mass")
        dt_value = float(dt if dt is not None else solid.contact_dynamic_dt)
        damping_value = float(damping if damping is not None else solid.contact_dynamic_damping)
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if not np.isfinite(damping_value) or damping_value < 0.0:
            raise ValueError("damping must be finite and >= 0")

        mass_full = np.asarray(solid.contact_dynamic_mass, dtype=np.float64).reshape(3 * solid.n_nodes)
        if np.any(~np.isfinite(mass_full)) or np.any(mass_full <= 0.0):
            raise ValueError("contact_dynamic_mass must be finite and positive")

        active_constraints: np.ndarray | None = None
        friction_slip_state: dict[tuple[int, int], float] = {}
        friction_trial_direction: dict[tuple[int, int], float] = {}
        total_iterations = 0
        result: SolidFEAResult | None = None
        system: Dict[str, np.ndarray | csr_matrix] | None = None
        final_constraints: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray]] | None = None
        final_solid = solid
        previous_free: np.ndarray | None = None
        x_prev: np.ndarray | None = None
        x_curr: np.ndarray | None = None

        for step_idx in range(1, steps + 1):
            alpha = float(step_idx) / float(steps)
            remapped_partner = cls._remapped_contact_partner(solid, result.u if result is not None else None)
            step_solid = cls._scaled_contact_step_solid(solid, alpha, contact_partner=remapped_partner)
            constraints = cls._contact_constraint_rows(step_solid)
            if not constraints:
                raise ValueError("dynamic contact solve requires at least one non-fixed contact candidate")
            if active_constraints is None:
                active_constraints = np.zeros((len(constraints),), dtype=bool)
            elif len(constraints) != int(active_constraints.size):
                raise ValueError("contact partner remap changed the active-set row count")
            final_constraints = constraints
            final_solid = step_solid

            base_system = cls.build_system(step_solid)
            free = np.asarray(base_system["free"], dtype=np.int64)
            if previous_free is None:
                previous_free = free.copy()
                x_prev = np.zeros((int(free.size),), dtype=np.float64)
                x_curr = np.zeros((int(free.size),), dtype=np.float64)
            elif not np.array_equal(previous_free, free):
                raise ValueError("dynamic contact solve requires a fixed free-DOF set")
            if x_prev is None or x_curr is None:
                raise ValueError("dynamic contact state was not initialized")

            mass_free = mass_full[free]
            dynamic_diag = mass_free / (dt_value * dt_value) + damping_value * mass_free / dt_value
            dynamic_kff = base_system["Kff"] + diags(dynamic_diag, offsets=0, format="csr")
            dynamic_rhs = np.asarray(base_system["Ff"], dtype=np.float64).copy()
            dynamic_rhs += (mass_free / (dt_value * dt_value)) * (2.0 * x_curr - x_prev)
            dynamic_rhs += (damping_value * mass_free / dt_value) * x_curr

            result, system = cls._solve_with_contact_constraints(
                step_solid,
                constraints,
                active_constraints,
                friction_slip_state=friction_slip_state,
                system_override=base_system,
                kff_override=dynamic_kff,
                ff_override=dynamic_rhs,
            )
            step_converged = False
            for _iteration in range(int(max_active_set_iter) + 1):
                total_iterations += 1
                signed_gap = cls._contact_signed_gaps(result.u, constraints)
                violated = signed_gap < -float(violation_tol)
                new_active = active_constraints | violated
                newly_active = new_active & ~active_constraints
                if not np.array_equal(new_active, active_constraints):
                    cls._initialize_contact_friction_state(
                        step_solid,
                        constraints,
                        newly_active,
                        result.u,
                        friction_slip_state,
                        friction_trial_direction,
                    )
                    active_constraints = new_active
                    result, system = cls._solve_with_contact_constraints(
                        step_solid,
                        constraints,
                        active_constraints,
                        friction_slip_state=friction_slip_state,
                        system_override=base_system,
                        kff_override=dynamic_kff,
                        ff_override=dynamic_rhs,
                    )
                    continue
                if cls._contact_friction_slip_updates(
                    step_solid,
                    system,
                    friction_slip_state,
                    friction_trial_direction,
                ):
                    result, system = cls._solve_with_contact_constraints(
                        step_solid,
                        constraints,
                        active_constraints,
                        friction_slip_state=friction_slip_state,
                        system_override=base_system,
                        kff_override=dynamic_kff,
                        ff_override=dynamic_rhs,
                    )
                    continue
                step_converged = True
                break
            if not step_converged:
                cls._attach_contact_friction_state(
                    step_solid,
                    constraints,
                    active_constraints,
                    friction_slip_state,
                    system,
                )
                system["contact_dynamic_steps"] = np.asarray([steps], dtype=np.int64)
                system["contact_dynamic_dt"] = np.asarray([dt_value], dtype=np.float64)
                system["contact_dynamic_damping"] = np.asarray([damping_value], dtype=np.float64)
                system["contact_dynamic_mass"] = mass_full.reshape(solid.n_nodes, 3).copy()
                system["contact_partner_initial"] = (
                    np.asarray(solid.contact_partner, dtype=np.int64).copy()
                    if solid.contact_partner is not None
                    else np.full((solid.n_nodes,), -1, dtype=np.int64)
                )
                system["contact_partner_final"] = (
                    np.asarray(step_solid.contact_partner, dtype=np.int64).copy()
                    if step_solid.contact_partner is not None
                    else np.full((step_solid.n_nodes,), -1, dtype=np.int64)
                )
                return SolidContactResult(
                    fea_result=result,
                    active_contact=cls._active_contact_mask(step_solid, constraints, active_constraints),
                    iterations=int(total_iterations),
                    converged=False,
                    active_solid=step_solid,
                    system=system,
                )

            solved_free = np.asarray(result.u, dtype=np.float64).reshape(3 * solid.n_nodes)[free].copy()
            x_prev, x_curr = x_curr.copy(), solved_free

        if result is None or system is None or active_constraints is None or final_constraints is None:
            raise ValueError("dynamic contact solve did not run any steps")
        cls._attach_contact_friction_state(
            final_solid,
            final_constraints,
            active_constraints,
            friction_slip_state,
            system,
        )
        system["contact_dynamic_steps"] = np.asarray([steps], dtype=np.int64)
        system["contact_dynamic_dt"] = np.asarray([dt_value], dtype=np.float64)
        system["contact_dynamic_damping"] = np.asarray([damping_value], dtype=np.float64)
        system["contact_dynamic_mass"] = mass_full.reshape(solid.n_nodes, 3).copy()
        system["contact_partner_initial"] = (
            np.asarray(solid.contact_partner, dtype=np.int64).copy()
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        system["contact_partner_final"] = (
            np.asarray(final_solid.contact_partner, dtype=np.int64).copy()
            if final_solid.contact_partner is not None
            else np.full((final_solid.n_nodes,), -1, dtype=np.int64)
        )
        return SolidContactResult(
            fea_result=result,
            active_contact=cls._active_contact_mask(final_solid, final_constraints, active_constraints),
            iterations=int(total_iterations),
            converged=True,
            active_solid=final_solid,
            system=system,
        )

    @classmethod
    def solve_contact_penalty_nonlinear(
        cls,
        solid: Solid,
        *,
        max_newton_iter: int = 30,
        residual_tol: float = 1.0e-9,
        step_tol: float = 1.0e-10,
    ) -> SolidContactResult:
        """
        Solve a bounded nonlinear penalty contact pilot.

        Contact rows use ``lambda = k * max(-gap - C u, 0) ** exponent``.
        Optional prescribed slip applies a Coulomb-limit tangential force from
        the same nonlinear normal multiplier. Optional inferred friction uses a
        tangent penalty response while sticking and clamps at ``mu * lambda``
        when slipping. This gives a small synthetic nonlinear contact law that
        remains target-free replayable from the exported stiffness and contact
        metadata. It is not a full finite-deformation or general frictional
        algorithm.
        """
        solid.validate()
        if max_newton_iter < 0:
            raise ValueError("max_newton_iter must be >= 0")
        if solid.contact_penalty_stiffness is None or solid.contact_penalty_exponent is None:
            raise ValueError("nonlinear penalty contact requires penalty metadata")
        if solid.contact_tangent_stick is not None:
            raise ValueError("nonlinear penalty contact currently supports normal contact plus slip/friction metadata only")

        constraints = cls._contact_constraint_rows(solid)
        if not constraints:
            raise ValueError("nonlinear penalty contact solve requires at least one contact candidate")

        system = cls.build_system(solid)
        K = system["K"]
        F = np.asarray(system["F"], dtype=np.float64)
        Kff: csr_matrix = system["Kff"]
        Ff = np.asarray(system["Ff"], dtype=np.float64)
        free = np.asarray(system["free"], dtype=np.int64)
        u_prescribed = np.asarray(system["u_prescribed"], dtype=np.float64).copy()
        full_to_free = np.full((3 * solid.n_nodes,), -1, dtype=np.int64)
        full_to_free[free] = np.arange(int(free.size), dtype=np.int64)
        penalty = np.asarray(solid.contact_penalty_stiffness, dtype=np.float64).reshape(solid.n_nodes)
        exponent = float(solid.contact_penalty_exponent)
        slip = (
            np.asarray(solid.contact_tangent_slip, dtype=np.float64)
            if solid.contact_tangent_slip is not None
            else np.zeros((solid.n_nodes, 3), dtype=np.float64)
        )
        friction = (
            np.asarray(solid.contact_tangent_friction, dtype=bool)
            if solid.contact_tangent_friction is not None
            else np.zeros((solid.n_nodes, 3), dtype=bool)
        )
        friction_mu = (
            np.asarray(solid.contact_friction_mu, dtype=np.float64)
            if solid.contact_friction_mu is not None
            else np.zeros((solid.n_nodes,), dtype=np.float64)
        )
        partner = (
            np.asarray(solid.contact_partner, dtype=np.int64)
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )

        try:
            x = np.asarray(spsolve(Kff, Ff), dtype=np.float64)
        except Exception:
            x = np.zeros((int(free.size),), dtype=np.float64)
        if x.shape != (int(free.size),) or np.any(~np.isfinite(x)):
            x = np.zeros((int(free.size),), dtype=np.float64)

        active_constraints = np.zeros((len(constraints),), dtype=bool)
        contact_force_flat = np.zeros((3 * solid.n_nodes,), dtype=np.float64)
        penetration_by_node = np.zeros((solid.n_nodes,), dtype=np.float64)
        converged = False
        iterations = 0

        for iteration in range(int(max_newton_iter) + 1):
            iterations = int(iteration)
            full = u_prescribed.copy()
            full[free] = x
            residual = np.asarray(Kff @ x, dtype=np.float64).reshape(-1) - Ff
            contact_force_flat = np.zeros((3 * solid.n_nodes,), dtype=np.float64)
            penetration_by_node = np.zeros((solid.n_nodes,), dtype=np.float64)
            active_constraints = np.zeros((len(constraints),), dtype=bool)
            jac_rows: list[int] = []
            jac_cols: list[int] = []
            jac_values: list[float] = []

            for constraint_idx, (dofs, coeffs, gap_value, node_id, _comps) in enumerate(constraints):
                signed_gap = float(gap_value + coeffs @ full[dofs])
                if signed_gap >= 0.0:
                    continue
                node_id = int(node_id)
                k_value = float(penalty[node_id])
                if k_value <= 0.0:
                    continue
                active_constraints[int(constraint_idx)] = True
                penetration = -signed_gap
                penetration_by_node[node_id] = max(float(penetration_by_node[node_id]), float(penetration))
                lambda_value = k_value * float(penetration**exponent)
                tangent_value = k_value * exponent * float(penetration ** (exponent - 1.0))
                normal_free_idx: list[int] = []
                normal_free_coeffs: list[float] = []
                force_free_idx: list[int] = []
                force_free_coeffs: list[float] = []
                for dof, coeff in zip(dofs, coeffs):
                    contact_force_flat[int(dof)] += lambda_value * float(coeff)
                    idx = int(full_to_free[int(dof)])
                    if idx >= 0:
                        normal_free_idx.append(idx)
                        normal_free_coeffs.append(float(coeff))
                        force_free_idx.append(idx)
                        force_free_coeffs.append(float(coeff))
                partner_id = int(partner[node_id])
                slip_norm = float(np.linalg.norm(slip[node_id]))
                if partner_id >= 0 and slip_norm > 1.0e-12 and float(friction_mu[node_id]) > 0.0:
                    tangent = slip[node_id] / slip_norm
                    tangent_comps = np.flatnonzero(np.abs(tangent) > 1.0e-12)
                    tangent_dofs = np.concatenate(
                        [
                            3 * node_id + tangent_comps.astype(np.int64, copy=True),
                            3 * partner_id + tangent_comps.astype(np.int64, copy=True),
                        ]
                    )
                    tangent_coeffs = np.concatenate([tangent[tangent_comps], -tangent[tangent_comps]])
                    tangent_coeffs = -float(friction_mu[node_id]) * tangent_coeffs.astype(np.float64)
                    for dof, coeff in zip(tangent_dofs, tangent_coeffs):
                        contact_force_flat[int(dof)] += lambda_value * float(coeff)
                        idx = int(full_to_free[int(dof)])
                        if idx >= 0:
                            force_free_idx.append(idx)
                            force_free_coeffs.append(float(coeff))
                if partner_id >= 0 and float(friction_mu[node_id]) > 0.0:
                    for comp in np.flatnonzero(friction[node_id]):
                        tangent_dofs = np.asarray([3 * node_id + int(comp), 3 * partner_id + int(comp)], dtype=np.int64)
                        tangent_coeffs = np.asarray([1.0, -1.0], dtype=np.float64)
                        tangent_gap = float(tangent_coeffs @ full[tangent_dofs])
                        if abs(tangent_gap) <= 1.0e-14:
                            continue
                        limit = float(friction_mu[node_id]) * lambda_value
                        trial_force = -tangent_value * tangent_gap
                        tangent_free_idx: list[int] = []
                        tangent_free_coeffs: list[float] = []
                        for dof, coeff in zip(tangent_dofs, tangent_coeffs):
                            idx = int(full_to_free[int(dof)])
                            if idx >= 0:
                                tangent_free_idx.append(idx)
                                tangent_free_coeffs.append(float(coeff))
                        if abs(trial_force) <= limit:
                            for dof, coeff in zip(tangent_dofs, tangent_coeffs):
                                contact_force_flat[int(dof)] += trial_force * float(coeff)
                            if tangent_free_idx:
                                tangent_idx_arr = np.asarray(tangent_free_idx, dtype=np.int64)
                                tangent_coeff_arr = np.asarray(tangent_free_coeffs, dtype=np.float64)
                                residual[tangent_idx_arr] -= trial_force * tangent_coeff_arr
                                outer_t = tangent_value * np.outer(tangent_coeff_arr, tangent_coeff_arr)
                                rr_t, cc_t = np.meshgrid(tangent_idx_arr, tangent_idx_arr, indexing="ij")
                                jac_rows.extend(int(v) for v in rr_t.reshape(-1))
                                jac_cols.extend(int(v) for v in cc_t.reshape(-1))
                                jac_values.extend(float(v) for v in outer_t.reshape(-1))
                        else:
                            slip_sign = 1.0 if tangent_gap > 0.0 else -1.0
                            tangent_force_coeffs = -float(friction_mu[node_id]) * slip_sign * tangent_coeffs
                            for dof, coeff in zip(tangent_dofs, tangent_force_coeffs):
                                contact_force_flat[int(dof)] += lambda_value * float(coeff)
                                idx = int(full_to_free[int(dof)])
                                if idx >= 0:
                                    force_free_idx.append(idx)
                                    force_free_coeffs.append(float(coeff))
                if not normal_free_idx or not force_free_idx:
                    continue
                normal_idx_arr = np.asarray(normal_free_idx, dtype=np.int64)
                normal_coeff_arr = np.asarray(normal_free_coeffs, dtype=np.float64)
                force_idx_arr = np.asarray(force_free_idx, dtype=np.int64)
                force_coeff_arr = np.asarray(force_free_coeffs, dtype=np.float64)
                residual[force_idx_arr] -= lambda_value * force_coeff_arr
                outer = tangent_value * np.outer(force_coeff_arr, normal_coeff_arr)
                rr, cc = np.meshgrid(force_idx_arr, normal_idx_arr, indexing="ij")
                jac_rows.extend(int(v) for v in rr.reshape(-1))
                jac_cols.extend(int(v) for v in cc.reshape(-1))
                jac_values.extend(float(v) for v in outer.reshape(-1))

            scale = max(1.0, float(np.linalg.norm(Ff)), float(np.linalg.norm(contact_force_flat[free])))
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= float(residual_tol) * scale:
                converged = True
                break
            jacobian = Kff
            if jac_values:
                jacobian = jacobian + coo_matrix(
                    (
                        np.asarray(jac_values, dtype=np.float64),
                        (np.asarray(jac_rows, dtype=np.int64), np.asarray(jac_cols, dtype=np.int64)),
                    ),
                    shape=Kff.shape,
                    dtype=np.float64,
                ).tocsr()
            dx = np.asarray(spsolve(jacobian, -residual), dtype=np.float64)
            if dx.shape != x.shape or np.any(~np.isfinite(dx)):
                break
            x = x + dx
            if float(np.linalg.norm(dx)) <= float(step_tol) * max(1.0, float(np.linalg.norm(x))):
                converged = True
                break

        full = u_prescribed.copy()
        full[free] = x
        contact_force_flat = np.zeros((3 * solid.n_nodes,), dtype=np.float64)
        penetration_by_node = np.zeros((solid.n_nodes,), dtype=np.float64)
        penalty_friction_slip_direction = np.zeros((solid.n_nodes, 3), dtype=np.float64)
        penalty_friction_stick = np.zeros((solid.n_nodes, 3), dtype=bool)
        active_constraints = np.zeros((len(constraints),), dtype=bool)
        for constraint_idx, (dofs, coeffs, gap_value, node_id, _comps) in enumerate(constraints):
            signed_gap = float(gap_value + coeffs @ full[dofs])
            if signed_gap >= 0.0:
                continue
            node_id = int(node_id)
            k_value = float(penalty[node_id])
            if k_value <= 0.0:
                continue
            active_constraints[int(constraint_idx)] = True
            penetration = -signed_gap
            penetration_by_node[node_id] = max(float(penetration_by_node[node_id]), float(penetration))
            lambda_value = k_value * float(penetration**exponent)
            tangent_value = k_value * exponent * float(penetration ** (exponent - 1.0))
            for dof, coeff in zip(dofs, coeffs):
                contact_force_flat[int(dof)] += lambda_value * float(coeff)
            partner_id = int(partner[node_id])
            slip_norm = float(np.linalg.norm(slip[node_id]))
            if partner_id >= 0 and slip_norm > 1.0e-12 and float(friction_mu[node_id]) > 0.0:
                tangent = slip[node_id] / slip_norm
                tangent_comps = np.flatnonzero(np.abs(tangent) > 1.0e-12)
                tangent_dofs = np.concatenate(
                    [
                        3 * node_id + tangent_comps.astype(np.int64, copy=True),
                        3 * partner_id + tangent_comps.astype(np.int64, copy=True),
                    ]
                )
                tangent_coeffs = np.concatenate([tangent[tangent_comps], -tangent[tangent_comps]])
                tangent_coeffs = -float(friction_mu[node_id]) * tangent_coeffs.astype(np.float64)
                for dof, coeff in zip(tangent_dofs, tangent_coeffs):
                    contact_force_flat[int(dof)] += lambda_value * float(coeff)
            if partner_id >= 0 and float(friction_mu[node_id]) > 0.0:
                for comp in np.flatnonzero(friction[node_id]):
                    tangent_dofs = np.asarray([3 * node_id + int(comp), 3 * partner_id + int(comp)], dtype=np.int64)
                    tangent_coeffs = np.asarray([1.0, -1.0], dtype=np.float64)
                    tangent_gap = float(tangent_coeffs @ full[tangent_dofs])
                    if abs(tangent_gap) <= 1.0e-14:
                        continue
                    limit = float(friction_mu[node_id]) * lambda_value
                    trial_force = -tangent_value * tangent_gap
                    if abs(trial_force) <= limit:
                        penalty_friction_stick[node_id, int(comp)] = True
                        for dof, coeff in zip(tangent_dofs, tangent_coeffs):
                            contact_force_flat[int(dof)] += trial_force * float(coeff)
                    else:
                        slip_sign = 1.0 if tangent_gap > 0.0 else -1.0
                        penalty_friction_slip_direction[node_id, int(comp)] = slip_sign
                        tangent_force_coeffs = -float(friction_mu[node_id]) * slip_sign * tangent_coeffs
                        for dof, coeff in zip(tangent_dofs, tangent_force_coeffs):
                            contact_force_flat[int(dof)] += lambda_value * float(coeff)
        reactions = (K @ full - F - contact_force_flat).reshape(solid.n_nodes, 3)
        u = full.reshape(solid.n_nodes, 3)
        system = dict(system)
        system["contact_penalty_stiffness"] = penalty.copy()
        system["contact_penalty_exponent"] = np.asarray([exponent], dtype=np.float64)
        system["contact_penalty_force"] = contact_force_flat.reshape(solid.n_nodes, 3).copy()
        system["contact_penalty_penetration"] = penetration_by_node.copy()
        system["contact_penalty_iterations"] = np.asarray([iterations], dtype=np.int64)
        system["contact_penalty_friction_slip_direction"] = penalty_friction_slip_direction
        system["contact_penalty_friction_stick"] = penalty_friction_stick
        return SolidContactResult(
            fea_result=cls.postprocess_fields(solid, u, reactions=reactions),
            active_contact=cls._active_contact_mask(solid, constraints, active_constraints),
            iterations=int(iterations),
            converged=bool(converged),
            active_solid=solid,
            system=system,
        )

    @classmethod
    def solve_contact_penalty_nonlinear_incremental(
        cls,
        solid: Solid,
        *,
        n_steps: int | None = None,
        max_newton_iter: int = 30,
        residual_tol: float = 1.0e-9,
        step_tol: float = 1.0e-10,
    ) -> SolidContactResult:
        """
        Replay nonlinear penalty contact over a proportional finite-sliding ramp.

        Each step scales external loads and prescribed displacements, optionally
        remaps paired contact partners from the previous deformed tangential
        position, then solves the same nonlinear normal-penalty and inferred
        Coulomb friction law as ``solve_contact_penalty_nonlinear``. This is a
        bounded synthetic finite-sliding pilot, not a full finite-deformation
        continuum contact solver.
        """
        solid.validate()
        steps = int(n_steps if n_steps is not None else (solid.contact_incremental_steps or 1))
        if steps <= 1 and not bool(solid.contact_partner_remap):
            return cls.solve_contact_penalty_nonlinear(
                solid,
                max_newton_iter=max_newton_iter,
                residual_tol=residual_tol,
                step_tol=step_tol,
            )
        if steps < 1:
            raise ValueError("n_steps must be >= 1")

        total_iterations = 0
        final_contact: SolidContactResult | None = None
        result: SolidFEAResult | None = None
        final_solid = solid
        converged = True

        for step_idx in range(1, steps + 1):
            alpha = float(step_idx) / float(steps)
            remapped_partner = cls._remapped_contact_partner(solid, result.u if result is not None else None)
            step_solid = cls._scaled_contact_step_solid(solid, alpha, contact_partner=remapped_partner)
            contact = cls.solve_contact_penalty_nonlinear(
                step_solid,
                max_newton_iter=max_newton_iter,
                residual_tol=residual_tol,
                step_tol=step_tol,
            )
            total_iterations += int(contact.iterations)
            final_contact = contact
            result = contact.fea_result
            final_solid = step_solid
            if not contact.converged:
                converged = False
                break

        if final_contact is None:
            raise ValueError("incremental nonlinear penalty contact solve did not run any steps")

        system = dict(final_contact.system)
        system["contact_incremental_steps"] = np.asarray([steps], dtype=np.int64)
        system["contact_partner_initial"] = (
            np.asarray(solid.contact_partner, dtype=np.int64).copy()
            if solid.contact_partner is not None
            else np.full((solid.n_nodes,), -1, dtype=np.int64)
        )
        system["contact_partner_final"] = (
            np.asarray(final_solid.contact_partner, dtype=np.int64).copy()
            if final_solid.contact_partner is not None
            else np.full((final_solid.n_nodes,), -1, dtype=np.int64)
        )
        system["contact_penalty_iterations"] = np.asarray([total_iterations], dtype=np.int64)
        return SolidContactResult(
            fea_result=final_contact.fea_result,
            active_contact=final_contact.active_contact,
            iterations=int(total_iterations),
            converged=bool(converged and final_contact.converged),
            active_solid=final_solid,
            system=system,
        )

    @classmethod
    def solve_axis_contact(
        cls,
        solid: Solid,
        *,
        max_active_set_iter: int = 20,
        violation_tol: float = 1.0e-10,
    ) -> SolidContactResult:
        """
        Solve a small-strain axis-aligned unilateral contact pilot problem.

        Contact fields use the convention ``gap + normal * u >= 0`` per
        candidate DOF. Active contacts are converted to essential BCs
        ``u = -gap / normal``. The active set is monotone-additive; this is
        adequate for the synthetic push-into-plane pilot and deliberately not a
        general frictional/contact algorithm.
        """
        return cls.solve_contact(
            solid,
            max_active_set_iter=max_active_set_iter,
            violation_tol=violation_tol,
        )

    @classmethod
    def solve(cls, solid: Solid) -> SolidFEAResult:
        solid.validate()
        sys = cls.build_system(solid)
        K = sys["K"]
        F = sys["F"]
        free = sys["free"]
        Kff = sys["Kff"]
        Ff = sys["Ff"]
        u_prescribed = sys["u_prescribed"]

        uf = spsolve(Kff, Ff)
        if np.any(~np.isfinite(uf)):
            raise ValueError("Linear solve failed (non-finite displacement result)")

        u = np.asarray(u_prescribed, dtype=np.float64).copy()
        u[free] = uf
        reactions = (K @ u - F).reshape(solid.n_nodes, 3)
        u = u.reshape(solid.n_nodes, 3)

        return cls.postprocess_fields(solid, u, reactions=reactions)

    @classmethod
    def postprocess_fields(
        cls,
        solid: Solid,
        u: np.ndarray,
        *,
        reactions: np.ndarray | None = None,
    ) -> SolidFEAResult:
        """Compute element and quadrature FE-state fields for a known displacement solution."""
        solid.validate()
        u = np.asarray(u, dtype=np.float64).reshape(solid.n_nodes, 3)
        if reactions is None:
            reactions = np.zeros((solid.n_nodes, 3), dtype=np.float64)
        else:
            reactions = np.asarray(reactions, dtype=np.float64).reshape(solid.n_nodes, 3)

        vm_list: list[float] = []
        energy_list: list[float] = []
        gauss_strain_list: list[np.ndarray] = []
        gauss_stress_list: list[np.ndarray] = []
        gauss_vm_list: list[np.ndarray] = []
        gauss_energy_list: list[np.ndarray] = []
        gauss_weight_volume_list: list[np.ndarray] = []

        def postprocess_element(elem_nodes: np.ndarray, E: float, nu: float) -> None:
            x = solid.nodes[elem_nodes]
            nper = int(elem_nodes.shape[0])
            ue = np.empty((3 * nper,), dtype=np.float64)
            for i, nid in enumerate(elem_nodes):
                ue[3 * i : 3 * i + 3] = u[int(nid)]

            strain, stress, vm_gp, energy_gp, weight_vol = cls._element_gauss_state(x=x, ue=ue, E=E, nu=nu, nper=nper)
            vm = float(np.mean(vm_gp))
            energy = float(np.sum(energy_gp))
            vm_list.append(vm)
            energy_list.append(energy)
            gauss_strain_list.append(strain)
            gauss_stress_list.append(stress)
            gauss_vm_list.append(vm_gp)
            gauss_energy_list.append(energy_gp)
            gauss_weight_volume_list.append(weight_vol)

        for eidx, elem in enumerate(solid.elements_c3d8):
            postprocess_element(elem, E=float(solid.E_c3d8[eidx]), nu=float(solid.nu_c3d8[eidx]))
        for eidx, elem in enumerate(solid.elements_c3d6):
            postprocess_element(elem, E=float(solid.E_c3d6[eidx]), nu=float(solid.nu_c3d6[eidx]))

        return SolidFEAResult(
            u=u,
            reactions=reactions,
            elem_vm_stress=np.asarray(vm_list, dtype=np.float64),
            elem_strain_energy=np.asarray(energy_list, dtype=np.float64),
            gauss_strain=np.vstack(gauss_strain_list) if gauss_strain_list else np.zeros((0, 6), dtype=np.float64),
            gauss_stress=np.vstack(gauss_stress_list) if gauss_stress_list else np.zeros((0, 6), dtype=np.float64),
            gauss_vm_stress=np.vstack(gauss_vm_list) if gauss_vm_list else np.zeros((0, 1), dtype=np.float64),
            gauss_strain_energy=np.vstack(gauss_energy_list) if gauss_energy_list else np.zeros((0, 1), dtype=np.float64),
            gauss_weight_volume=np.vstack(gauss_weight_volume_list) if gauss_weight_volume_list else np.zeros((0, 1), dtype=np.float64),
        )


# Backward-compatible helper API
solve_solid = LinearSolidSolver.solve
