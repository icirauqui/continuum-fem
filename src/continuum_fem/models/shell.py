from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Shell:
    """Thin-shell plate model with S4 (quad) and S3 (tri) elements.

    Nodal generalized DOFs follow:
      - dof 0: translation ux
      - dof 1: translation uy
      - dof 2: translation uz
      - dof 3: rotation rx
      - dof 4: rotation ry
      - dof 5: rotation rz
    """

    nodes: np.ndarray
    elements_s4: np.ndarray
    elements_s3: np.ndarray
    t_s4: np.ndarray
    E_s4: np.ndarray
    nu_s4: np.ndarray
    t_s3: np.ndarray
    E_s3: np.ndarray
    nu_s3: np.ndarray
    loads: np.ndarray
    fixed: np.ndarray

    def validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (N, 3)")
        n_nodes = int(self.nodes.shape[0])

        if self.loads.shape != (n_nodes, 6):
            raise ValueError("loads must have shape (N, 6)")
        if self.fixed.shape != (n_nodes, 6):
            raise ValueError("fixed must have shape (N, 6)")
        if self.fixed.dtype != bool:
            raise ValueError("fixed must be bool array")

        if self.elements_s4.ndim != 2 or self.elements_s4.shape[1] != 4:
            raise ValueError("elements_s4 must have shape (E4, 4)")
        if self.elements_s3.ndim != 2 or self.elements_s3.shape[1] != 3:
            raise ValueError("elements_s3 must have shape (E3, 3)")

        e4 = int(self.elements_s4.shape[0])
        e3 = int(self.elements_s3.shape[0])

        if self.t_s4.shape != (e4,):
            raise ValueError("t_s4 must have shape (E4,)")
        if self.E_s4.shape != (e4,):
            raise ValueError("E_s4 must have shape (E4,)")
        if self.nu_s4.shape != (e4,):
            raise ValueError("nu_s4 must have shape (E4,)")

        if self.t_s3.shape != (e3,):
            raise ValueError("t_s3 must have shape (E3,)")
        if self.E_s3.shape != (e3,):
            raise ValueError("E_s3 must have shape (E3,)")
        if self.nu_s3.shape != (e3,):
            raise ValueError("nu_s3 must have shape (E3,)")

        if np.any(self.t_s4 <= 0.0) or np.any(self.t_s3 <= 0.0):
            raise ValueError("Shell thickness must be > 0")
        if np.any(self.E_s4 <= 0.0) or np.any(self.E_s3 <= 0.0):
            raise ValueError("Young's modulus must be > 0")

        if np.any(self.nu_s4 <= -0.99) or np.any(self.nu_s4 >= 0.49):
            raise ValueError("nu_s4 values must be in (-0.99, 0.49)")
        if np.any(self.nu_s3 <= -0.99) or np.any(self.nu_s3 >= 0.49):
            raise ValueError("nu_s3 values must be in (-0.99, 0.49)")

        for name, elems, width in (
            ("S4", self.elements_s4, 4),
            ("S3", self.elements_s3, 3),
        ):
            if elems.shape[1] != width:
                raise ValueError(f"{name} elements must have width {width}")
            if elems.size == 0:
                continue
            if np.any(elems < 0) or np.any(elems >= n_nodes):
                raise ValueError(f"{name} elements contain out-of-range node indices")
            ordered = np.sort(elems, axis=1)
            repeated = np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
            if np.any(repeated):
                raise ValueError(f"{name} element has repeated node indices")

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_s4(self) -> int:
        return int(self.elements_s4.shape[0])

    @property
    def n_s3(self) -> int:
        return int(self.elements_s3.shape[0])

    @property
    def n_elements(self) -> int:
        return self.n_s4 + self.n_s3
