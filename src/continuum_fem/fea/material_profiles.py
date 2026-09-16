"""Explicit source-derived, study-conditioned four-fiber material profiles.

The adjacent JSON is the canonical coefficient/evidence catalog and must ship
with this module. These are fixed author-derived candidates, not population or
viscoelastic tissue truth. Bulk is always a caller assignment in Pa. By default
the numerical material uses Pa; supplying stress_unit_Pa constructs a scaled
material with ALL stresslike coefficients divided by that unit. Geometry and
loads are never rescaled here. Retain source_profile_config with every run.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from .four_fiber import FourFiberMaterial

_CATALOG_BYTES = Path(__file__).with_suffix('.json').read_bytes()
_CATALOG = json.loads(_CATALOG_BYTES)
_CATALOG_SHA256 = hashlib.sha256(_CATALOG_BYTES).hexdigest()
_STRESS_FIELDS = ('matrix_coefficient', 'axial_k1',
                  'circumferential_k1', 'diagonal_k1', 'bulk')


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number, not a boolean or string')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and strictly positive')
    return value


def list_source_profiles() -> tuple[str, ...]:
    """Return the three fixed source specimen IDs; no sampling distribution."""
    return tuple(_CATALOG['profiles'])


def source_profile_metadata(profile_id: str) -> dict:
    """Return independent JSON-serializable provenance and evidence metadata."""
    if profile_id not in _CATALOG['profiles']:
        raise KeyError(f'Unknown source material profile {profile_id!r}; '
                       f'choose one of {list_source_profiles()}')
    return deepcopy({
        'profile_id': profile_id,
        'schema_version': _CATALOG['schema_version'],
        'catalog_id': _CATALOG['catalog_id'],
        'catalog_sha256': _CATALOG_SHA256,
        'common': _CATALOG['common'],
        'profile': _CATALOG['profiles'][profile_id],
        'evidence_source_hashes': _CATALOG['evidence_source_hashes'],
    })


def source_profile_config(profile_id: str, *, bulk_pa: float,
                          stress_unit_Pa: float = 1.0) -> dict:
    """Prepare reproducible constructor inputs and explicit unit assignment.

    stress_unit_Pa=1 uses numeric SI stress. For a nondimensional solver supply
    its named physical stress unit explicitly; k2 and angles remain unchanged.
    No default bulk, constitutive refit, or implicit solver-unit inference.
    """
    metadata = source_profile_metadata(profile_id)
    bulk_pa = _positive_real(bulk_pa, 'bulk_pa')
    stress_unit_Pa = _positive_real(stress_unit_Pa, 'stress_unit_Pa')
    physical = deepcopy(metadata['profile']['material_parameters_without_bulk'])
    physical['bulk'] = bulk_pa
    converted = deepcopy(physical)
    for name in _STRESS_FIELDS:
        converted[name] /= stress_unit_Pa
        if not math.isfinite(converted[name]) or converted[name] <= 0:
            raise ValueError(f'Unit conversion makes {name} nonfinite or nonpositive')
    return {
        'profile_id': profile_id,
        'assigned_bulk_pa': bulk_pa,
        'stress_unit_Pa': stress_unit_Pa,
        'stress_conversion': 'solver_value = physical_Pa / stress_unit_Pa',
        'physical_parameters_Pa': physical,
        'material_parameters': converted,
        'provenance': metadata,
    }


@dataclass(frozen=True)
class SourceProfileMaterial(FourFiberMaterial):
    """Same tensor evaluator with accurate metadata for explicit stress units."""
    source_profile_id: str = ''
    stress_unit_Pa: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        _positive_real(self.stress_unit_Pa, 'stress_unit_Pa')
        expected = source_profile_config(
            self.source_profile_id, bulk_pa=self.bulk * self.stress_unit_Pa,
            stress_unit_Pa=self.stress_unit_Pa)['material_parameters']
        for name, value in expected.items():
            actual = getattr(self, name)
            agrees = (actual == value if isinstance(value, str) else
                      math.isclose(actual, value, rel_tol=1e-14, abs_tol=0))
            if not agrees:
                raise ValueError(f'{name} differs from the fixed source profile')

    @property
    def convention_metadata(self):
        metadata = dict(super().convention_metadata)
        metadata.update({
            'coefficient_units': 'solver stress units; k2 dimensionless; theta radians',
            'stress_unit_Pa': float(self.stress_unit_Pa),
            'stress_conversion': 'physical_Pa = solver_value * stress_unit_Pa',
            'source_profile_id': self.source_profile_id,
            'catalog_sha256': _CATALOG_SHA256,
            'bulk_empirically_identified': False,
        })
        return metadata


def build_source_material(profile_id: str, *, bulk_pa: float,
                          stress_unit_Pa: float = 1.0) -> SourceProfileMaterial:
    """Build a fixed source profile; assigned bulk and stress units are explicit."""
    config = source_profile_config(profile_id, bulk_pa=bulk_pa,
                                   stress_unit_Pa=stress_unit_Pa)
    return SourceProfileMaterial(**config['material_parameters'],
                                 source_profile_id=profile_id,
                                 stress_unit_Pa=config['stress_unit_Pa'])
