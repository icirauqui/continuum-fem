import numpy as np

from continuum_fem.fea.solid import LinearSolidSolver
from continuum_fem.systems.solid_builder import SolidBuilder
from continuum_fem.systems.tube_c3d6_response import solve_hollow_cylinder_c3d6_force_response


def test_hollow_c3d6_tube_has_clamped_ends_and_a_finite_linear_response():
    solid = SolidBuilder.build_hollow_cylinder_c3d6(
        length=4.0, inner_radius=0.8, outer_radius=1.0, n_axial=4, n_theta=8,
        E=1.0e6, nu=0.30,
    )
    assert solid.n_c3d8 == 0
    assert solid.n_c3d6 == 64
    response = LinearSolidSolver.solve(solid)
    assert np.isfinite(response.u).all()


def test_tube_force_response_is_linear_and_preserves_the_declared_grid_shape():
    kwargs = dict(
        length=4.0, inner_radius=0.8, outer_radius=1.0, n_axial=4, n_theta=8,
        young_modulus_pa=1.0e6, poisson_ratio=0.30, center_s=0.5, center_theta=0.25,
    )
    unit = np.array([0.0, 0.0, 125.0])
    one = solve_hollow_cylinder_c3d6_force_response(**kwargs, force_vector=unit)
    two = solve_hollow_cylinder_c3d6_force_response(**kwargs, force_vector=2.0 * unit)
    reverse = solve_hollow_cylinder_c3d6_force_response(**kwargs, force_vector=-unit)
    assert one.displacement.shape == (5, 8, 3)
    assert np.isfinite(one.displacement).all()
    np.testing.assert_allclose(two.displacement, 2.0 * one.displacement, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(reverse.displacement, -one.displacement, rtol=1e-8, atol=1e-12)
