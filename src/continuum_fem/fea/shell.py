from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from ..models.shell import Shell


@dataclass(slots=True)
class ShellFEAResult:
    """Result container for linear shell solve."""

    u: np.ndarray
    reactions: np.ndarray
    elem_vm_stress: np.ndarray
    elem_strain_energy: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "u": self.u,
            "reactions": self.reactions,
            "elem_vm_stress": self.elem_vm_stress,
            "elem_strain_energy": self.elem_strain_energy,
        }


class LinearShellSolver:
    """Linear 6-DOF shell solver for S4 and S3 elements.

    Nodal DOFs:
      [ux, uy, uz, rx, ry, rz]

    Element formulation:
      - membrane: plane-stress in local shell plane
      - bending: Mindlin-Reissner
      - transverse shear: Mindlin-Reissner
      - drilling rotation rz: small stabilization term
    """

    DETJ_TOL = 1e-12
    PLANAR_ELEM_TOL = 1e-7
    SHEAR_CORR = 5.0 / 6.0
    DRILL_STAB = 1.0e-6
    # T3 Mindlin elements can lock strongly; reduced effective shear improves practical behavior.
    S3_SHEAR_SCALE = 1.0e-2

    _GAUSS_2 = (
        (-1.0 / np.sqrt(3.0), 1.0),
        (1.0 / np.sqrt(3.0), 1.0),
    )

    @staticmethod
    def _Dm(E: float, nu: float, t: float) -> np.ndarray:
        fac = E * t / (1.0 - nu * nu)
        return fac * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _Db(E: float, nu: float, t: float) -> np.ndarray:
        fac = E * (t**3) / (12.0 * (1.0 - nu * nu))
        return fac * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _Ds(cls, E: float, nu: float, t: float) -> np.ndarray:
        g = E / (2.0 * (1.0 + nu))
        return (cls.SHEAR_CORR * g * t) * np.eye(2, dtype=np.float64)

    @staticmethod
    def _Dps(E: float, nu: float) -> np.ndarray:
        fac = E / (1.0 - nu * nu)
        return fac * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _B_membrane(dndx: np.ndarray, dndy: np.ndarray) -> np.ndarray:
        n = int(dndx.size)
        B = np.zeros((3, 6 * n), dtype=np.float64)
        for i in range(n):
            j = 6 * i
            B[0, j + 0] = dndx[i]
            B[1, j + 1] = dndy[i]
            B[2, j + 0] = dndy[i]
            B[2, j + 1] = dndx[i]
        return B

    @staticmethod
    def _B_bending(dndx: np.ndarray, dndy: np.ndarray) -> np.ndarray:
        n = int(dndx.size)
        B = np.zeros((3, 6 * n), dtype=np.float64)
        for i in range(n):
            j = 6 * i
            B[0, j + 3] = dndx[i]
            B[1, j + 4] = dndy[i]
            B[2, j + 3] = dndy[i]
            B[2, j + 4] = dndx[i]
        return B

    @staticmethod
    def _B_shear(N: np.ndarray, dndx: np.ndarray, dndy: np.ndarray) -> np.ndarray:
        n = int(dndx.size)
        B = np.zeros((2, 6 * n), dtype=np.float64)
        for i in range(n):
            j = 6 * i
            B[0, j + 2] = dndx[i]
            B[0, j + 3] = -N[i]
            B[1, j + 2] = dndy[i]
            B[1, j + 4] = -N[i]
        return B

    @staticmethod
    def _local_axes(x: np.ndarray) -> np.ndarray:
        v1 = x[1] - x[0]
        v2 = x[2] - x[0]
        n1 = float(np.linalg.norm(v1))
        n2 = np.cross(v1, v2)
        n3 = float(np.linalg.norm(n2))
        if n1 <= 1e-14 or n3 <= 1e-14:
            raise ValueError("Degenerate shell element geometry.")

        e1 = v1 / n1
        e3 = n2 / n3
        e2 = np.cross(e3, e1)
        e2 /= max(float(np.linalg.norm(e2)), 1e-14)
        return np.vstack([e1, e2, e3]).astype(np.float64)

    @classmethod
    def _to_local_xy(cls, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = cls._local_axes(x)
        d = x - x[0]
        xloc = d @ R[0, :]
        yloc = d @ R[1, :]
        zloc = d @ R[2, :]
        if float(np.max(np.abs(zloc))) > cls.PLANAR_ELEM_TOL:
            raise ValueError("Shell element nodes are not coplanar within tolerance.")
        xy = np.column_stack([xloc, yloc]).astype(np.float64)
        return xy, R

    @staticmethod
    def _make_T(n_nodes: int, R: np.ndarray) -> np.ndarray:
        T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=np.float64)
        for i in range(n_nodes):
            j = 6 * i
            T[j : j + 3, j : j + 3] = R
            T[j + 3 : j + 6, j + 3 : j + 6] = R
        return T

    @staticmethod
    def _polygon_area(xy: np.ndarray) -> float:
        x = xy[:, 0]
        y = xy[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _s4_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ],
            dtype=np.float64,
        )
        dndxi = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=np.float64,
        )
        dndeta = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=np.float64,
        )
        return N, dndxi, dndeta

    @classmethod
    def _s4_dndx_dndy(
        cls,
        xy: np.ndarray,
        xi: float,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        N, dndxi, dndeta = cls._s4_shape(xi, eta)
        x = xy[:, 0]
        y = xy[:, 1]
        J = np.array(
            [
                [float(np.dot(dndxi, x)), float(np.dot(dndxi, y))],
                [float(np.dot(dndeta, x)), float(np.dot(dndeta, y))],
            ],
            dtype=np.float64,
        )
        detJ = float(np.linalg.det(J))
        if abs(detJ) <= cls.DETJ_TOL:
            raise ValueError("Invalid S4 Jacobian (detJ too small).")
        invJ = np.linalg.inv(J)
        dndx = invJ[0, 0] * dndxi + invJ[0, 1] * dndeta
        dndy = invJ[1, 0] * dndxi + invJ[1, 1] * dndeta
        return N, dndx, dndy, detJ

    @classmethod
    def _s3_geom(cls, xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x1, y1 = float(xy[0, 0]), float(xy[0, 1])
        x2, y2 = float(xy[1, 0]), float(xy[1, 1])
        x3, y3 = float(xy[2, 0]), float(xy[2, 1])
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * area2
        if abs(area) <= cls.DETJ_TOL:
            raise ValueError("Invalid S3 geometry (zero area).")
        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
        dndx = b / (2.0 * area)
        dndy = c / (2.0 * area)
        return abs(area), dndx, dndy

    @classmethod
    def _ke_s4_local(cls, xy: np.ndarray, E: float, nu: float, t: float) -> np.ndarray:
        Dm = cls._Dm(E, nu, t)
        Db = cls._Db(E, nu, t)
        Ds = cls._Ds(E, nu, t)

        ke = np.zeros((24, 24), dtype=np.float64)

        for xi, wx in cls._GAUSS_2:
            for eta, wy in cls._GAUSS_2:
                N, dndx, dndy, detJ = cls._s4_dndx_dndy(xy, xi, eta)
                wdet = abs(detJ) * wx * wy
                Bm = cls._B_membrane(dndx, dndy)
                Bb = cls._B_bending(dndx, dndy)
                ke += (Bm.T @ Dm @ Bm + Bb.T @ Db @ Bb) * wdet

        N0, dndx0, dndy0, detJ0 = cls._s4_dndx_dndy(xy, 0.0, 0.0)
        Bs = cls._B_shear(N0, dndx0, dndy0)
        ke += (Bs.T @ Ds @ Bs) * abs(detJ0) * 4.0

        area = max(cls._polygon_area(xy), 1e-16)
        kdr = cls.DRILL_STAB * E * t * area / 4.0
        for i in range(4):
            ke[6 * i + 5, 6 * i + 5] += kdr

        return ke

    @classmethod
    def _ke_s3_local(cls, xy: np.ndarray, E: float, nu: float, t: float) -> np.ndarray:
        area, dndx, dndy = cls._s3_geom(xy)
        Dm = cls._Dm(E, nu, t)
        Db = cls._Db(E, nu, t)
        Ds = cls._Ds(E, nu, t)

        N_cent = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        Bm = cls._B_membrane(dndx, dndy)
        Bb = cls._B_bending(dndx, dndy)
        Bs = cls._B_shear(N_cent, dndx, dndy)

        ke = (
            Bm.T @ Dm @ Bm
            + Bb.T @ Db @ Bb
            + cls.S3_SHEAR_SCALE * (Bs.T @ Ds @ Bs)
        ) * area

        kdr = cls.DRILL_STAB * E * t * area / 3.0
        for i in range(3):
            ke[6 * i + 5, 6 * i + 5] += kdr

        return ke

    @classmethod
    def _element_stiffness_global(
        cls,
        xg: np.ndarray,
        E: float,
        nu: float,
        t: float,
        nper: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy, R = cls._to_local_xy(xg)
        if nper == 4:
            kel = cls._ke_s4_local(xy, E=E, nu=nu, t=t)
        elif nper == 3:
            kel = cls._ke_s3_local(xy, E=E, nu=nu, t=t)
        else:
            raise ValueError(f"Unsupported shell element node count: {nper}")
        T = cls._make_T(nper, R)
        keg = T.T @ kel @ T
        return keg, kel, T

    @classmethod
    def assemble_global_K(cls, shell: Shell) -> csr_matrix:
        shell.validate()

        n_dofs = 6 * shell.n_nodes
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def push_element(elem_nodes: np.ndarray, E: float, nu: float, t: float) -> None:
            xg = shell.nodes[elem_nodes]
            nper = int(elem_nodes.size)
            keg, _, _ = cls._element_stiffness_global(xg=xg, E=E, nu=nu, t=t, nper=nper)

            dofs = np.empty((6 * nper,), dtype=np.int64)
            for i, nid in enumerate(elem_nodes):
                b = 6 * int(nid)
                dofs[6 * i : 6 * i + 6] = np.arange(b, b + 6, dtype=np.int64)

            rr = np.repeat(dofs, dofs.size)
            cc = np.tile(dofs, dofs.size)
            rows.append(rr)
            cols.append(cc)
            vals.append(keg.reshape(-1))

        for eidx, elem in enumerate(shell.elements_s4):
            push_element(
                elem_nodes=elem,
                E=float(shell.E_s4[eidx]),
                nu=float(shell.nu_s4[eidx]),
                t=float(shell.t_s4[eidx]),
            )

        for eidx, elem in enumerate(shell.elements_s3):
            push_element(
                elem_nodes=elem,
                E=float(shell.E_s3[eidx]),
                nu=float(shell.nu_s3[eidx]),
                t=float(shell.t_s3[eidx]),
            )

        if not rows:
            raise ValueError("Shell has no elements to assemble.")

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(vals)
        return coo_matrix((data, (row, col)), shape=(n_dofs, n_dofs), dtype=np.float64).tocsr()

    @classmethod
    def build_system(cls, shell: Shell) -> Dict[str, np.ndarray | csr_matrix]:
        K = cls.assemble_global_K(shell)
        F = shell.loads.reshape(-1).astype(np.float64)
        fixed_mask = shell.fixed.reshape(-1)
        free = np.where(~fixed_mask)[0]
        if free.size == 0:
            raise ValueError("All shell DOFs are fixed.")

        Kff = K[free][:, free].tocsr()
        Ff = F[free]
        return {"K": K, "F": F, "free": free, "Kff": Kff, "Ff": Ff}

    @classmethod
    def _element_vm_local(
        cls,
        xy: np.ndarray,
        ue_local: np.ndarray,
        E: float,
        nu: float,
        t: float,
        nper: int,
    ) -> float:
        Dps = cls._Dps(E, nu)
        if nper == 4:
            N, dndx, dndy, _ = cls._s4_dndx_dndy(xy, 0.0, 0.0)
        else:
            _area, dndx, dndy = cls._s3_geom(xy)
            N = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

        Bm = cls._B_membrane(dndx, dndy)
        Bb = cls._B_bending(dndx, dndy)
        Bs = cls._B_shear(N, dndx, dndy)

        eps_m = Bm @ ue_local
        kappa = Bb @ ue_local
        strain_top = eps_m + 0.5 * t * kappa
        stress = Dps @ strain_top

        # Include transverse shear contribution as equivalent distortion component.
        shear = Bs @ ue_local
        tau_xz, tau_yz = float(shear[0]), float(shear[1])

        sxx, syy, txy = [float(v) for v in stress]
        vm = np.sqrt(
            sxx * sxx
            - sxx * syy
            + syy * syy
            + 3.0 * (txy * txy + tau_xz * tau_xz + tau_yz * tau_yz)
        )
        return float(vm)

    @classmethod
    def solve(cls, shell: Shell) -> ShellFEAResult:
        shell.validate()
        system = cls.build_system(shell)

        K = system["K"]
        F = system["F"]
        free = system["free"]
        Kff = system["Kff"]
        Ff = system["Ff"]

        uf = spsolve(Kff, Ff)
        if np.any(~np.isfinite(uf)):
            raise np.linalg.LinAlgError("Shell solve returned non-finite values.")

        u_flat = np.zeros((6 * shell.n_nodes,), dtype=np.float64)
        u_flat[free] = np.asarray(uf, dtype=np.float64)
        u = u_flat.reshape(shell.n_nodes, 6)

        reactions_flat = (K @ u_flat) - F
        reactions = reactions_flat.reshape(shell.n_nodes, 6)

        vm: list[float] = []
        energy: list[float] = []

        for eidx, elem in enumerate(shell.elements_s4):
            nper = 4
            dofs = np.array([6 * int(nid) + d for nid in elem for d in range(6)], dtype=np.int64)
            ue_g = u_flat[dofs]
            xg = shell.nodes[elem]
            E = float(shell.E_s4[eidx])
            nu = float(shell.nu_s4[eidx])
            t = float(shell.t_s4[eidx])
            _keg, kel, T = cls._element_stiffness_global(xg=xg, E=E, nu=nu, t=t, nper=nper)
            ue_l = T @ ue_g
            xy, _R = cls._to_local_xy(xg)
            vm.append(cls._element_vm_local(xy=xy, ue_local=ue_l, E=E, nu=nu, t=t, nper=nper))
            energy.append(float(0.5 * ue_l @ (kel @ ue_l)))

        for eidx, elem in enumerate(shell.elements_s3):
            nper = 3
            dofs = np.array([6 * int(nid) + d for nid in elem for d in range(6)], dtype=np.int64)
            ue_g = u_flat[dofs]
            xg = shell.nodes[elem]
            E = float(shell.E_s3[eidx])
            nu = float(shell.nu_s3[eidx])
            t = float(shell.t_s3[eidx])
            _keg, kel, T = cls._element_stiffness_global(xg=xg, E=E, nu=nu, t=t, nper=nper)
            ue_l = T @ ue_g
            xy, _R = cls._to_local_xy(xg)
            vm.append(cls._element_vm_local(xy=xy, ue_local=ue_l, E=E, nu=nu, t=t, nper=nper))
            energy.append(float(0.5 * ue_l @ (kel @ ue_l)))

        return ShellFEAResult(
            u=u,
            reactions=reactions,
            elem_vm_stress=np.asarray(vm, dtype=np.float64),
            elem_strain_energy=np.asarray(energy, dtype=np.float64),
        )


def solve_shell(shell: Shell) -> Dict[str, np.ndarray]:
    return LinearShellSolver.solve(shell).as_dict()
