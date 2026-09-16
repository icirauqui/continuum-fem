import numpy as np

from continuum_fem.fea.hgo import HGOMaterial
from continuum_fem.fea.hyperelastic import FiberFamily, HyperelasticMaterial
from continuum_fem.fea.invariant_polynomial import PolynomialFiberFamily, PolynomialInvariantMaterial


def test_hyperelastic_material_is_stress_free_at_the_reference_state():
    material = HyperelasticMaterial(mu=2.0, bulk=100.0, fibers=(FiberFamily(direction=(1.0, 0.0, 0.0), ksi=1.0, alpha=1.0, beta=3.0),))
    state = material.evaluate(np.eye(3))
    np.testing.assert_allclose(state.first_piola, np.zeros((3, 3)), atol=1e-12)


def test_hgo_and_invariant_polynomial_models_evaluate_finite_states():
    deformation = np.diag([1.05, 1.0 / np.sqrt(1.05), 1.0 / np.sqrt(1.05)])
    hgo = HGOMaterial(mu=1.0, bulk=100.0, k1=0.5, k2=2.0, theta=0.5)
    polynomial = PolynomialInvariantMaterial(
        matrix_coefficients=(0.5, 0.0, 0.0), bulk=100.0,
        fibers=(PolynomialFiberFamily(direction=(1.0, 0.0, 0.0), coefficients=(0.2, 0.1, 0.0)),),
    )
    assert np.isfinite(hgo.evaluate(deformation).first_piola).all()
    assert np.isfinite(polynomial.evaluate(deformation).first_piola).all()
