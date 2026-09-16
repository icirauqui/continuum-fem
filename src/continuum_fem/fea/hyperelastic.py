"""Finite-strain constitutive pilot, independent of the existing linear solver.

Energies per reference volume follow Maas et al. (2012), doi:10.1115/1.4005694:
Eq. (11), coupled neo-Hookean; Eq. (9), C2=0, split neo-Hookean.
The optional full-I4 fiber family is the tension-only exponential-power law
specified in FEBio's official manual (fiber-exp-pow), with shear term disabled,
unit recruitment stretch. beta>=3 gives a C2 energy at activation; beta=2
has a one-sided tangent (inactive branch at exact recruitment). These are
assigned material laws, not calibrated colon tissue or a finite-element solver.

F[i,J]=dx_i/dX_J; P[i,J]=dW/dF[i,J]; A[i,J,k,L]=dP[i,J]/dF[k,L].
mu, bulk, ksi, W and stresses share pressure units. F, J, alpha, beta are
unitless. All stored derivatives are full tensors, not engineering Voigt arrays.
Near-incompressible bulk/mu ratios need a separately verified mixed/locking-free
FE discretization. No incompressibility constraint, history, viscosity or contact
is implemented here. Inputs may have leading batch dimensions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class FiberFamily:
    direction: tuple[float, float, float]
    ksi: float
    alpha: float = 1.0
    beta: float = 3.0

    def __post_init__(self):
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-12):
            raise ValueError('Fiber reference direction must be a finite unit vector')
        if not np.isfinite([self.ksi, self.alpha, self.beta]).all() or self.ksi <= 0 or self.alpha < 0 or self.beta < 2:
            raise ValueError('Fiber parameters require finite ksi>0, alpha>=0, beta>=2')
        object.__setattr__(self, 'direction', tuple(float(x) for x in direction))


@dataclass(frozen=True)
class HyperelasticState:
    deformation_gradient: np.ndarray
    jacobian: np.ndarray
    energy_density: np.ndarray
    first_piola: np.ndarray
    second_piola: np.ndarray
    cauchy: np.ndarray
    green_lagrange: np.ndarray
    tangent_dP_dF: np.ndarray


@dataclass(frozen=True)
class HyperelasticMaterial:
    """Stable-ground-matrix pilot with optional additive oriented fibers.

    split: W=mu/2*(J**(-2/3)*tr(F.T F)-3)+bulk/2*log(J)**2.
    coupled: W=mu/2*(tr(F.T F)-3)-mu*log(J)+lambda/2*log(J)**2,
             lambda=bulk-2*mu/3. This pilot restricts lambda>=0.
    fiber: W=ksi/(alpha*beta)*expm1(alpha*max(I4-1,0)**beta),
           using ksi/beta*max(I4-1,0)**beta for alpha=0.

    Fiber I4 uses the full F, not its isochoric part. Finite bulk includes
    volumetric penalty; it does not impose exact incompressibility. beta=2
    has a tangent discontinuity at I4=1: this API chooses the inactive tangent
    there; derivatives must be checked away from recruitment or one-sided.
    """
    mu: float
    bulk: float
    form: Literal['split', 'coupled'] = 'split'
    fibers: tuple[FiberFamily, ...] = ()
    minimum_jacobian: float = 1e-12

    def __post_init__(self):
        if not np.isfinite([self.mu, self.bulk, self.minimum_jacobian]).all() or min(self.mu, self.bulk, self.minimum_jacobian) <= 0:
            raise ValueError('Require finite positive mu, bulk and minimum_jacobian')
        if self.form not in {'split', 'coupled'}:
            raise ValueError('form must be split or coupled')
        if self.form == 'coupled' and self.bulk < 2*self.mu/3:
            raise ValueError('Coupled pilot requires nonnegative Lame lambda: bulk>=2*mu/3')
        if not all(isinstance(f, FiberFamily) for f in self.fibers):
            raise TypeError('fibers must contain FiberFamily objects')
        object.__setattr__(self, 'fibers', tuple(self.fibers))

    def evaluate(self, deformation_gradient: np.ndarray) -> HyperelasticState:
        F = np.array(deformation_gradient, dtype=np.float64, copy=True)
        if F.shape[-2:] != (3, 3) or not np.isfinite(F).all():
            raise ValueError('F must be finite with shape (...,3,3)')
        J = np.linalg.det(F)
        if np.any(J <= self.minimum_jacobian) or not np.isfinite(J).all():
            raise ValueError('F has an inverted, singular or inadmissibly small positive Jacobian')
        H = np.swapaxes(np.linalg.inv(F), -1, -2)  # F^{-T}
        logJ = np.log(J)
        I1 = np.sum(F*F, axis=(-2, -1))
        identity = np.eye(3)
        identity4 = np.einsum('ik,JL->iJkL', identity, identity)
        hh = np.einsum('...iJ,...kL->...iJkL', H, H)
        inverse_derivative = np.einsum('...iL,...kJ->...iJkL', H, H)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                if self.form == 'coupled':
                    lam = self.bulk-2*self.mu/3
                    coefficient = lam*logJ-self.mu
                    W = .5*self.mu*(I1-3)-self.mu*logJ+.5*lam*logJ**2
                    P = self.mu*F+coefficient[..., None, None]*H
                    A = self.mu*identity4+lam*hh-coefficient[..., None, None, None, None]*inverse_derivative
                else:
                    scale = J**(-2/3)
                    deviator = F-I1[..., None, None]/3*H
                    W = .5*self.mu*(scale*I1-3)+.5*self.bulk*logJ**2
                    P = self.mu*scale[..., None, None]*deviator+self.bulk*logJ[..., None, None]*H
                    A = self.mu*scale[..., None, None, None, None]*(identity4
                        -(2/3)*np.einsum('...iJ,...kL->...iJkL', deviator, H)
                        -(2/3)*np.einsum('...iJ,...kL->...iJkL', H, F)
                        +I1[..., None, None, None, None]/3*inverse_derivative)
                    A += self.bulk*hh-self.bulk*logJ[..., None, None, None, None]*inverse_derivative
                for fiber in self.fibers:
                    direction = np.asarray(fiber.direction)
                    current = F @ direction
                    extension = np.maximum(np.sum(current*current, axis=-1)-1., 0.)
                    power = extension**fiber.beta
                    exponent = fiber.alpha*power
                    exponential = np.exp(exponent)
                    # Exact exprel form avoids ksi/alpha overflowing as alpha
                    # approaches zero although the power-law limit is finite.
                    exprel=np.divide(np.expm1(exponent),exponent,out=np.ones_like(exponent),where=exponent!=0)
                    W += (fiber.ksi/fiber.beta)*power*exprel
                    first = fiber.ksi*extension**(fiber.beta-1)*exponential
                    second = np.where(extension>0, fiber.ksi*extension**(fiber.beta-2)*(fiber.beta-1+fiber.alpha*fiber.beta*power)*exponential, 0.)
                    dyad = np.einsum('...i,J->...iJ', current, direction)
                    P += 2*first[..., None, None]*dyad
                    A += 4*second[..., None, None, None, None]*np.einsum('...iJ,...kL->...iJkL', dyad, dyad)
                    A += 2*first[..., None, None, None, None]*np.einsum('ik,J,L->iJkL', identity, direction, direction)
                second_piola = np.swapaxes(H, -1, -2) @ P
                cauchy = P @ np.swapaxes(F, -1, -2)/J[..., None, None]
                green = .5*(np.swapaxes(F, -1, -2) @ F-identity)
        except FloatingPointError as exc:
            raise ValueError('Constitutive evaluation overflowed; state outside numerical admissibility') from exc
        if not all(np.isfinite(x).all() for x in (W, P, A, second_piola, cauchy, green)):
            raise ValueError('Nonfinite constitutive output')
        return HyperelasticState(F, J, W, P, second_piola, cauchy, green, A)
