# Architecture

ContinuumFEM separates three concerns.

`models` contains explicit dataclasses for meshes, material parameters, loads,
and constraints. `systems` builds deterministic parametric instances. `fea`
assembles and solves them. This boundary prevents application code, surrogate
models, and experiment policy from becoming implicit solver dependencies.

The public API is namespaced as `continuum_fem`. Consumers should not import
internal files by path or add this repository to `sys.path` as `fea`/`systems`.
