from __future__ import annotations

import numpy as np

from ..models.shell import Shell


class ShellBuilder:
    """Factory helpers for structured shell meshes and cantilever setups.

    Nodal DOF order is:
      [ux, uy, uz, rx, ry, rz]
    and loads follow the generalized force order:
      [Fx, Fy, Fz, Mx, My, Mz]
    """

    @staticmethod
    def _node_index(i: int, j: int, ny: int) -> int:
        return i * (ny + 1) + j

    @staticmethod
    def _pipe_node_index(i: int, j: int, n_theta: int) -> int:
        return i * n_theta + j

    @staticmethod
    def _element_area_normal(nodes: np.ndarray, conn: np.ndarray) -> tuple[float, np.ndarray]:
        pts = nodes[np.asarray(conn, dtype=np.int64)]
        if pts.shape[0] == 3:
            a = pts[1] - pts[0]
            b = pts[2] - pts[0]
            nvec = np.cross(a, b)
            area = 0.5 * float(np.linalg.norm(nvec))
            if area <= 1e-16:
                return 0.0, np.zeros((3,), dtype=np.float64)
            return area, nvec / (2.0 * area)

        if pts.shape[0] == 4:
            a1 = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            a2 = np.cross(pts[2] - pts[0], pts[3] - pts[0])
            avec = 0.5 * (a1 + a2)
            area = float(np.linalg.norm(avec))
            if area <= 1e-16:
                return 0.0, np.zeros((3,), dtype=np.float64)
            return area, avec / area

        raise ValueError("conn must have length 3 or 4")

    @staticmethod
    def _angle_distance(theta: float, center: float) -> float:
        return float(np.arctan2(np.sin(theta - center), np.cos(theta - center)))

    @classmethod
    def build_rectangular_shell(
        cls,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        t: float,
        E: float,
        nu: float,
        element_type: str = "s4",
    ) -> Shell:
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be >= 1")
        if Lx <= 0.0 or Ly <= 0.0:
            raise ValueError("Lx and Ly must be > 0")
        if t <= 0.0:
            raise ValueError("t must be > 0")
        if E <= 0.0:
            raise ValueError("E must be > 0")
        if not (-0.99 < nu < 0.49):
            raise ValueError("nu must be in (-0.99, 0.49)")

        et = element_type.strip().lower()
        if et not in {"s3", "s4"}:
            raise ValueError("element_type must be 's3' or 's4'")

        xs = np.linspace(0.0, float(Lx), nx + 1, dtype=np.float64)
        ys = np.linspace(0.0, float(Ly), ny + 1, dtype=np.float64)

        nodes = np.zeros(((nx + 1) * (ny + 1), 3), dtype=np.float64)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                nid = cls._node_index(i, j, ny)
                nodes[nid, 0] = x
                nodes[nid, 1] = y
                nodes[nid, 2] = 0.0

        s4_list: list[list[int]] = []
        s3_list: list[list[int]] = []

        for i in range(nx):
            for j in range(ny):
                n00 = cls._node_index(i, j, ny)
                n10 = cls._node_index(i + 1, j, ny)
                n11 = cls._node_index(i + 1, j + 1, ny)
                n01 = cls._node_index(i, j + 1, ny)

                if et == "s4":
                    # Counterclockwise in XY plane.
                    s4_list.append([n00, n10, n11, n01])
                else:
                    # Two triangles with consistent orientation.
                    s3_list.append([n00, n10, n11])
                    s3_list.append([n00, n11, n01])

        elements_s4 = np.asarray(s4_list, dtype=np.int64) if s4_list else np.zeros((0, 4), dtype=np.int64)
        elements_s3 = np.asarray(s3_list, dtype=np.int64) if s3_list else np.zeros((0, 3), dtype=np.int64)

        e4 = int(elements_s4.shape[0])
        e3 = int(elements_s3.shape[0])

        shell = Shell(
            nodes=nodes,
            elements_s4=elements_s4,
            elements_s3=elements_s3,
            t_s4=np.full((e4,), float(t), dtype=np.float64),
            E_s4=np.full((e4,), float(E), dtype=np.float64),
            nu_s4=np.full((e4,), float(nu), dtype=np.float64),
            t_s3=np.full((e3,), float(t), dtype=np.float64),
            E_s3=np.full((e3,), float(E), dtype=np.float64),
            nu_s3=np.full((e3,), float(nu), dtype=np.float64),
            loads=np.zeros((nodes.shape[0], 6), dtype=np.float64),
            fixed=np.zeros((nodes.shape[0], 6), dtype=bool),
        )
        shell.validate()
        return shell

    @classmethod
    def build_cylindrical_shell(
        cls,
        nx: int,
        n_theta: int,
        L: float,
        R: float,
        t: float,
        E: float,
        nu: float,
        element_type: str = "s4",
    ) -> Shell:
        if nx < 1:
            raise ValueError("nx must be >= 1")
        if n_theta < 3:
            raise ValueError("n_theta must be >= 3")
        if L <= 0.0:
            raise ValueError("L must be > 0")
        if R <= 0.0:
            raise ValueError("R must be > 0")
        if t <= 0.0:
            raise ValueError("t must be > 0")
        if E <= 0.0:
            raise ValueError("E must be > 0")
        if not (-0.99 < nu < 0.49):
            raise ValueError("nu must be in (-0.99, 0.49)")

        et = element_type.strip().lower()
        if et not in {"s3", "s4"}:
            raise ValueError("element_type must be 's3' or 's4'")

        xs = np.linspace(0.0, float(L), nx + 1, dtype=np.float64)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)

        nodes = np.zeros(((nx + 1) * n_theta, 3), dtype=np.float64)
        for i, x in enumerate(xs):
            for j, th in enumerate(thetas):
                nid = cls._pipe_node_index(i, j, n_theta)
                nodes[nid, 0] = x
                nodes[nid, 1] = float(R * np.cos(th))
                nodes[nid, 2] = float(R * np.sin(th))

        s4_list: list[list[int]] = []
        s3_list: list[list[int]] = []
        for i in range(nx):
            for j in range(n_theta):
                jp = (j + 1) % n_theta
                n00 = cls._pipe_node_index(i, j, n_theta)
                n10 = cls._pipe_node_index(i, jp, n_theta)
                n11 = cls._pipe_node_index(i + 1, jp, n_theta)
                n01 = cls._pipe_node_index(i + 1, j, n_theta)

                if et == "s4":
                    s4_list.append([n00, n10, n11, n01])
                else:
                    s3_list.append([n00, n10, n11])
                    s3_list.append([n00, n11, n01])

        elements_s4 = np.asarray(s4_list, dtype=np.int64) if s4_list else np.zeros((0, 4), dtype=np.int64)
        elements_s3 = np.asarray(s3_list, dtype=np.int64) if s3_list else np.zeros((0, 3), dtype=np.int64)

        e4 = int(elements_s4.shape[0])
        e3 = int(elements_s3.shape[0])

        shell = Shell(
            nodes=nodes,
            elements_s4=elements_s4,
            elements_s3=elements_s3,
            t_s4=np.full((e4,), float(t), dtype=np.float64),
            E_s4=np.full((e4,), float(E), dtype=np.float64),
            nu_s4=np.full((e4,), float(nu), dtype=np.float64),
            t_s3=np.full((e3,), float(t), dtype=np.float64),
            E_s3=np.full((e3,), float(E), dtype=np.float64),
            nu_s3=np.full((e3,), float(nu), dtype=np.float64),
            loads=np.zeros((nodes.shape[0], 6), dtype=np.float64),
            fixed=np.zeros((nodes.shape[0], 6), dtype=bool),
        )
        shell.validate()
        return shell

    @classmethod
    def make_cantilever_plate_problem(
        cls,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        t: float,
        E: float,
        nu: float,
        total_load_z: float,
        element_type: str = "s4",
        edge_top_only: bool = False,
        total_load_vec: np.ndarray | None = None,
        total_moment_vec: np.ndarray | None = None,
    ) -> Shell:
        shell = cls.build_rectangular_shell(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            t=t,
            E=E,
            nu=nu,
            element_type=element_type,
        )

        x = shell.nodes[:, 0]
        y = shell.nodes[:, 1]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        y_med = float(np.median(y))
        tol = max(1e-9, 1e-8 * max(1.0, abs(x_max - x_min)))

        fixed = np.zeros((shell.n_nodes, 6), dtype=bool)
        # Clamped left edge by default.
        fixed[np.abs(x - x_min) <= tol, :] = True

        loads = np.zeros((shell.n_nodes, 6), dtype=np.float64)
        right = np.where(np.abs(x - x_max) <= tol)[0]
        loaded = right
        if edge_top_only:
            loaded = right[y[right] >= y_med - 1e-12]
        if loaded.size == 0:
            loaded = right

        force = np.asarray(total_load_vec, dtype=np.float64).reshape(3) if total_load_vec is not None else np.zeros((3,), dtype=np.float64)
        moment = (
            np.asarray(total_moment_vec, dtype=np.float64).reshape(3)
            if total_moment_vec is not None
            else np.zeros((3,), dtype=np.float64)
        )
        force[2] += float(total_load_z)

        loads[loaded, 0:3] = force.reshape(1, 3) / max(1, loaded.size)
        loads[loaded, 3:6] = moment.reshape(1, 3) / max(1, loaded.size)

        shell.loads = loads
        shell.fixed = fixed
        shell.validate()
        return shell

    @classmethod
    def make_tension_plate_with_holes_problem(
        cls,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        t: float,
        E: float,
        nu: float,
        holes: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
        total_load_x: float,
        total_load_y: float = 0.0,
        total_load_z: float = 0.0,
        element_type: str = "s4",
    ) -> Shell:
        """Structured tensile plate with circular cutouts removed at element centers.

        The plate is clamped on the left edge and loaded on the right edge.
        Circular holes are represented by deleting structured cells whose
        centers lie inside any requested cutout, then remapping unused nodes.
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx and ny must be >= 2")
        if Lx <= 0.0 or Ly <= 0.0:
            raise ValueError("Lx and Ly must be > 0")
        if t <= 0.0:
            raise ValueError("t must be > 0")
        if E <= 0.0:
            raise ValueError("E must be > 0")
        if not (-0.99 < nu < 0.49):
            raise ValueError("nu must be in (-0.99, 0.49)")

        et = element_type.strip().lower()
        if et not in {"s3", "s4"}:
            raise ValueError("element_type must be 's3' or 's4'")

        hole_specs = [(float(cx), float(cy), float(r)) for cx, cy, r in holes]
        for cx, cy, radius in hole_specs:
            if radius <= 0.0:
                raise ValueError("hole radius must be > 0")
            if cx - radius <= 0.0 or cx + radius >= Lx:
                raise ValueError("hole must remain inside the left/right plate boundaries")
            if cy - radius <= 0.0 or cy + radius >= Ly:
                raise ValueError("hole must remain inside the bottom/top plate boundaries")

        xs = np.linspace(0.0, float(Lx), nx + 1, dtype=np.float64)
        ys = np.linspace(0.0, float(Ly), ny + 1, dtype=np.float64)
        all_nodes = np.zeros(((nx + 1) * (ny + 1), 3), dtype=np.float64)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                nid = cls._node_index(i, j, ny)
                all_nodes[nid, 0] = x
                all_nodes[nid, 1] = y
                all_nodes[nid, 2] = 0.0

        kept_cells: list[tuple[int, int, int, int]] = []
        for i in range(nx):
            for j in range(ny):
                cx_cell = 0.5 * (xs[i] + xs[i + 1])
                cy_cell = 0.5 * (ys[j] + ys[j + 1])
                inside_hole = any((cx_cell - hx) ** 2 + (cy_cell - hy) ** 2 <= hr**2 for hx, hy, hr in hole_specs)
                if inside_hole:
                    continue

                n00 = cls._node_index(i, j, ny)
                n10 = cls._node_index(i + 1, j, ny)
                n11 = cls._node_index(i + 1, j + 1, ny)
                n01 = cls._node_index(i, j + 1, ny)
                kept_cells.append((n00, n10, n11, n01))

        if not kept_cells:
            raise ValueError("hole configuration removed all elements")

        used = sorted({nid for cell in kept_cells for nid in cell})
        remap = {old: new for new, old in enumerate(used)}
        nodes = all_nodes[np.asarray(used, dtype=np.int64)]

        s4_list: list[list[int]] = []
        s3_list: list[list[int]] = []
        for n00, n10, n11, n01 in kept_cells:
            r00, r10, r11, r01 = remap[n00], remap[n10], remap[n11], remap[n01]
            if et == "s4":
                s4_list.append([r00, r10, r11, r01])
            else:
                s3_list.append([r00, r10, r11])
                s3_list.append([r00, r11, r01])

        elements_s4 = np.asarray(s4_list, dtype=np.int64) if s4_list else np.zeros((0, 4), dtype=np.int64)
        elements_s3 = np.asarray(s3_list, dtype=np.int64) if s3_list else np.zeros((0, 3), dtype=np.int64)
        e4 = int(elements_s4.shape[0])
        e3 = int(elements_s3.shape[0])

        fixed = np.zeros((nodes.shape[0], 6), dtype=bool)
        loads = np.zeros((nodes.shape[0], 6), dtype=np.float64)
        x = nodes[:, 0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        tol = max(1e-9, 1e-8 * max(1.0, abs(x_max - x_min)))
        fixed[np.abs(x - x_min) <= tol, :] = True
        loaded = np.where(np.abs(x - x_max) <= tol)[0]
        if loaded.size == 0:
            raise ValueError("right load edge has no retained nodes")
        total_load = np.asarray([total_load_x, total_load_y, total_load_z], dtype=np.float64)
        loads[loaded, 0:3] = total_load.reshape(1, 3) / float(loaded.size)

        shell = Shell(
            nodes=nodes,
            elements_s4=elements_s4,
            elements_s3=elements_s3,
            t_s4=np.full((e4,), float(t), dtype=np.float64),
            E_s4=np.full((e4,), float(E), dtype=np.float64),
            nu_s4=np.full((e4,), float(nu), dtype=np.float64),
            t_s3=np.full((e3,), float(t), dtype=np.float64),
            E_s3=np.full((e3,), float(E), dtype=np.float64),
            nu_s3=np.full((e3,), float(nu), dtype=np.float64),
            loads=loads,
            fixed=fixed,
        )
        shell.validate()
        return shell

    @classmethod
    def make_pipe_problem(
        cls,
        nx: int,
        n_theta: int,
        L: float,
        R: float,
        t: float,
        E: float,
        nu: float,
        pressure_inward: float,
        element_type: str = "s4",
        x_patch: tuple[float, float] = (0.65, 1.0),
        theta_center_deg: float = 90.0,
        theta_span_deg: float = 90.0,
        clamp_start: bool = True,
        clamp_end: bool = False,
        total_load_vec: np.ndarray | None = None,
        total_moment_vec: np.ndarray | None = None,
    ) -> Shell:
        shell = cls.build_cylindrical_shell(
            nx=nx,
            n_theta=n_theta,
            L=L,
            R=R,
            t=t,
            E=E,
            nu=nu,
            element_type=element_type,
        )

        nodes = shell.nodes
        x = nodes[:, 0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        tol = max(1e-9, 1e-8 * max(1.0, abs(x_max - x_min)))

        fixed = np.zeros((shell.n_nodes, 6), dtype=bool)
        if clamp_start:
            fixed[np.abs(x - x_min) <= tol, :] = True
        if clamp_end:
            fixed[np.abs(x - x_max) <= tol, :] = True

        loads = np.zeros((shell.n_nodes, 6), dtype=np.float64)
        x_lo = float(min(x_patch[0], x_patch[1])) * float(L)
        x_hi = float(max(x_patch[0], x_patch[1])) * float(L)
        theta_c = float(np.deg2rad(theta_center_deg))
        theta_half = 0.5 * abs(float(np.deg2rad(theta_span_deg)))

        if shell.n_s4 > 0:
            conns = shell.elements_s4
            w = 0.25
        else:
            conns = shell.elements_s3
            w = 1.0 / 3.0

        chosen: list[tuple[np.ndarray, float, np.ndarray]] = []
        for conn in conns:
            pts = nodes[np.asarray(conn, dtype=np.int64)]
            ctr = pts.mean(axis=0)
            if ctr[0] < x_lo or ctr[0] > x_hi:
                continue
            theta = float(np.arctan2(ctr[2], ctr[1]))
            if abs(cls._angle_distance(theta, theta_c)) > theta_half:
                continue

            area, nrm = cls._element_area_normal(nodes, np.asarray(conn, dtype=np.int64))
            if area <= 1e-14:
                continue
            chosen.append((np.asarray(conn, dtype=np.int64), area, nrm))

        if not chosen and len(conns) > 0:
            # Robust fallback: pick nearest element center when patch discretization is sparse.
            best_idx = -1
            best_score = float("inf")
            for eidx, conn in enumerate(conns):
                pts = nodes[np.asarray(conn, dtype=np.int64)]
                ctr = pts.mean(axis=0)
                theta = float(np.arctan2(ctr[2], ctr[1]))
                x_rel = abs((ctr[0] - 0.5 * (x_lo + x_hi)) / max(L, 1e-12))
                th_rel = abs(cls._angle_distance(theta, theta_c)) / max(theta_half, 1e-12)
                score = x_rel + th_rel
                if score < best_score:
                    best_score = score
                    best_idx = eidx
            if best_idx >= 0:
                conn = np.asarray(conns[best_idx], dtype=np.int64)
                area, nrm = cls._element_area_normal(nodes, conn)
                if area > 1e-14:
                    chosen.append((conn, area, nrm))

        for conn, area, nrm in chosen:
            f_vec = -float(pressure_inward) * area * nrm
            for nid in conn:
                loads[int(nid), 0:3] += w * f_vec

        force = (
            np.asarray(total_load_vec, dtype=np.float64).reshape(3)
            if total_load_vec is not None
            else np.zeros((3,), dtype=np.float64)
        )
        moment = (
            np.asarray(total_moment_vec, dtype=np.float64).reshape(3)
            if total_moment_vec is not None
            else np.zeros((3,), dtype=np.float64)
        )
        if np.linalg.norm(force) > 0.0 or np.linalg.norm(moment) > 0.0:
            loaded_nodes = np.where((x >= x_lo - 1e-12) & (x <= x_hi + 1e-12))[0]
            if loaded_nodes.size == 0:
                loaded_nodes = np.arange(shell.n_nodes)
            loads[loaded_nodes, 0:3] += force.reshape(1, 3) / float(loaded_nodes.size)
            loads[loaded_nodes, 3:6] += moment.reshape(1, 3) / float(loaded_nodes.size)

        shell.loads = loads
        shell.fixed = fixed
        shell.validate()
        return shell

    @staticmethod
    def rotate_shell_problem(shell: Shell, R: np.ndarray) -> Shell:
        """Rotate geometry/loads/reactions frame by a 3x3 rotation matrix.

        The returned shell is physically equivalent in the rotated frame.
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError("R must have shape (3, 3)")
        should_be_I = R @ R.T
        if not np.allclose(should_be_I, np.eye(3), atol=1e-8):
            raise ValueError("R must be orthonormal.")
        if np.linalg.det(R) <= 0.0:
            raise ValueError("R must be a proper rotation (det(R) > 0).")

        nodes_rot = shell.nodes @ R.T
        loads_rot = shell.loads.copy()
        loads_rot[:, 0:3] = shell.loads[:, 0:3] @ R.T
        loads_rot[:, 3:6] = shell.loads[:, 3:6] @ R.T

        out = Shell(
            nodes=nodes_rot,
            elements_s4=shell.elements_s4.copy(),
            elements_s3=shell.elements_s3.copy(),
            t_s4=shell.t_s4.copy(),
            E_s4=shell.E_s4.copy(),
            nu_s4=shell.nu_s4.copy(),
            t_s3=shell.t_s3.copy(),
            E_s3=shell.E_s3.copy(),
            nu_s3=shell.nu_s3.copy(),
            loads=loads_rot,
            fixed=shell.fixed.copy(),
        )
        out.validate()
        return out


def build_rectangular_shell(*args, **kwargs) -> Shell:
    return ShellBuilder.build_rectangular_shell(*args, **kwargs)


def build_cylindrical_shell(*args, **kwargs) -> Shell:
    return ShellBuilder.build_cylindrical_shell(*args, **kwargs)


def make_cantilever_plate_problem(*args, **kwargs) -> Shell:
    return ShellBuilder.make_cantilever_plate_problem(*args, **kwargs)


def make_tension_plate_with_holes_problem(*args, **kwargs) -> Shell:
    return ShellBuilder.make_tension_plate_with_holes_problem(*args, **kwargs)


def make_pipe_problem(*args, **kwargs) -> Shell:
    return ShellBuilder.make_pipe_problem(*args, **kwargs)


def rotate_shell_problem(*args, **kwargs) -> Shell:
    return ShellBuilder.rotate_shell_problem(*args, **kwargs)
