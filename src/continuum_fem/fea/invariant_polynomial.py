"""Bounded isochoric polynomial material-point pilot from Fahmy et al. (2024).

Energy family follows doi:10.3390/bioengineering11010064 equations4/5:
Wiso=sum(a_i*(Ibar1-3)^i), i=1..3;
Wfiber=sum(c_i*(Ibar4-1)^i), i=2..4, per reference fiber direction.
A finite bulk/2*ln(J)^2 penalty is an explicit added volumetric assumption.
No published uniaxial reduction is used. Negative polynomial coefficients can
cause instability outside a bounded calibration domain: this implementation
checks finite arithmetic/J, NOT global stability or biological validity.
Fibers are NOT tension-only in this energy family. Full tensor analytic
stress/tangent derivatives are derived from the energy; no engineeringVoigt.
"""
from dataclasses import dataclass
import numpy as np
from .hyperelastic import HyperelasticState


@dataclass(frozen=True)
class PolynomialFiberFamily:
    direction: tuple[float,float,float]
    coefficients: tuple[float,float,float]

    def __post_init__(self):
        a=np.asarray(self.direction,float);c=np.asarray(self.coefficients,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or not np.isclose(np.linalg.norm(a),1.,rtol=0,atol=1e-12):
            raise ValueError('Require finite unit reference fiber direction')
        if c.shape!=(3,) or not np.isfinite(c).all():raise ValueError('Require finite quadratic/cubic/quartic coefficients')
        object.__setattr__(self,'direction',tuple(float(x) for x in a))
        object.__setattr__(self,'coefficients',tuple(float(x) for x in c))


@dataclass(frozen=True)
class PolynomialInvariantMaterial:
    matrix_coefficients: tuple[float,float,float]
    bulk: float
    fibers: tuple[PolynomialFiberFamily,...]=()
    minimum_jacobian: float=1e-12

    def __post_init__(self):
        a=np.asarray(self.matrix_coefficients,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or a[0]<=0:raise ValueError('Require finite matrix coefficients and positive linear invariant coefficient')
        if not np.isfinite([self.bulk,self.minimum_jacobian]).all() or min(self.bulk,self.minimum_jacobian)<=0:raise ValueError('Require positive finite bulk/minimumJacobian')
        if not all(isinstance(f,PolynomialFiberFamily) for f in self.fibers):raise ValueError('Require PolynomialFiberFamily entries')
        object.__setattr__(self,'matrix_coefficients',tuple(float(x) for x in a))
        object.__setattr__(self,'fibers',tuple(self.fibers))

    @property
    def mu(self):return 2*self.matrix_coefficients[0]

    def evaluate(self,deformation_gradient):
        F=np.array(deformation_gradient,dtype=float,copy=True)
        if F.shape[-2:]!=(3,3) or not np.isfinite(F).all():raise ValueError('Expected finite (...,3,3) F')
        J=np.linalg.det(F)
        if np.any(J<=self.minimum_jacobian) or not np.isfinite(J).all():raise ValueError('Invalid deformation Jacobian')
        H=np.swapaxes(np.linalg.inv(F),-1,-2);q=J**(-2/3);eye=np.eye(3);logJ=np.log(J)
        HH=np.einsum('...iJ,...kL->...iJkL',H,H)
        Q=np.einsum('...iL,...kJ->...iJkL',H,H)
        W=.5*self.bulk*logJ**2;P=self.bulk*logJ[...,None,None]*H
        A=self.bulk*HH-self.bulk*logJ[...,None,None,None,None]*Q
        def add_term(invariant,dyad,base_hessian,offset,coefficients,powers):
            nonlocal W,P,A
            value=q*invariant-offset
            shape=dyad-invariant[...,None,None]/3*H
            gradient=2*q[...,None,None]*shape
            hessian=2*q[...,None,None,None,None]*(base_hessian
                -(2/3)*np.einsum('...iJ,...kL->...iJkL',shape,H)
                -(2/3)*np.einsum('...iJ,...kL->...iJkL',H,dyad)
                +invariant[...,None,None,None,None]/3*Q)
            energy=sum(c*value**n for c,n in zip(coefficients,powers))
            first=sum(n*c*value**(n-1) for c,n in zip(coefficients,powers))
            second=sum(n*(n-1)*c*value**(n-2) for c,n in zip(coefficients,powers) if n>1)
            W=W+energy;P=P+first[...,None,None]*gradient
            A=A+first[...,None,None,None,None]*hessian+second[...,None,None,None,None]*np.einsum('...iJ,...kL->...iJkL',gradient,gradient)
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                add_term(np.sum(F*F,axis=(-2,-1)),F,np.einsum('ik,JL->iJkL',eye,eye),3.,self.matrix_coefficients,(1,2,3))
                for fiber in self.fibers:
                    a=np.asarray(fiber.direction);current=F@a
                    add_term(np.sum(current*current,axis=-1),np.einsum('...i,J->...iJ',current,a),np.einsum('ik,J,L->iJkL',eye,a,a),1.,fiber.coefficients,(2,3,4))
                second_piola=np.swapaxes(H,-1,-2)@P
                cauchy=P@np.swapaxes(F,-1,-2)/J[...,None,None]
                green=.5*(np.swapaxes(F,-1,-2)@F-eye)
        except FloatingPointError as exc:raise ValueError('Polynomial constitutive overflow') from exc
        if not all(np.isfinite(x).all() for x in (W,P,A,second_piola,cauchy,green)):raise ValueError('Nonfinite polynomial state')
        return HyperelasticState(F,J,W,P,second_piola,cauchy,green,A)
