# ContinuumFEM

ContinuumFEM is a small, transparent Python library for deterministic finite
element experiments in continuum mechanics. It provides linear truss, shell,
and three-dimensional solid solvers, parametric system builders, and selected
nonlinear material models. It deliberately does not contain learned surrogates,
training pipelines, datasets, or application-specific rendering code.

The project was extracted from `mesh-to-graph` so that classical finite-element
models can be versioned, tested, and reused independently by research projects
such as LumenMorph.

## Included capabilities

- Linear truss, shell, C3D8, and C3D6 solid solves.
- Hollow C3D6 tube construction and a calibrated force-response helper.
- Parametric truss, shell, and solid builders.
- Hyperelastic, HGO, four-fiber, invariant-polynomial, finite-strain, and
  optional mixed-tetrahedral research components.

The library is a research tool. A successful solve is not by itself a validated
biomechanical or clinical model.

## Install

```bash
git clone https://github.com/icirauqui/continuum-fem.git
cd continuum-fem
uv sync --all-groups
uv run pytest -q
```

Python 3.12+ is supported. The optional mixed-tetrahedral implementation needs
the `mixed` extra:

```bash
uv sync --all-groups --extra mixed
```

## First tube solve

```python
import numpy as np
from continuum_fem.systems.tube_c3d6_response import solve_hollow_cylinder_c3d6_force_response

response = solve_hollow_cylinder_c3d6_force_response(
    length=0.20, inner_radius=0.010, outer_radius=0.012,
    n_axial=12, n_theta=24, young_modulus_pa=1.0e6,
    poisson_ratio=0.30, center_s=0.5, center_theta=0.25,
    force_vector=np.array([0.0, 0.0, 1.0]),
)
print(response.displacement.shape)  # (13, 24, 3)
```

See [`notebooks/`](notebooks/) for compact, runnable truss, shell, and hollow
tube examples. The exported displacement field is a deterministic numerical
result under the declared mesh, material, loads, and boundary conditions.

## Repository layout

```text
src/continuum_fem/
  fea/       assemblers, solvers, and material models
  models/    explicit physical-system data models
  systems/   deterministic parametric model builders
tests/       solver and contract tests
notebooks/   compact executable examples
```

## Provenance and license

This is a source-preserving extraction of the FEA core from
`mesh-to-graph` revision `3056ce219a72d0dd1bb62e34ec6c7f5fee25fd64`, with only
package-import changes required by the new namespace. The original code is
licensed under GPL-3.0-or-later, retained in [`LICENSE`](LICENSE).

## Contributing

Keep units, boundary conditions, material assumptions, convergence status, and
verification evidence explicit. Add a focused test for every numerical change;
do not replace a failed mechanics check with a visual demonstration.
