"""Holzapfel-Gasser-Ogden paired-fiber material, separate from older laws.

Original law: Holzapfel, Gasser & Ogden (2000), JElasticity61:1-48,
doi:10.1023/A:1010835316564, Eqs64/65, activation on p37.
Wiso=mu/2*(Ibar1-3)+sum k1/(2*k2)*expm1(k2*<Ibar4-1>_+**2).
mu is original c (twice the matrix coefficient called mu by Puertolas2020).
Two equivalent fibers at +/-theta radians from reference circumferential x.
A bulk/2*log(J)^2 penalty is OPTIONAL NUMERICAL EXTENSION, not the source's
exact incompressibility. evaluate_isochoric exposes the unpenalized state for
later mixed formulations. No tissue validation, history, contact or failure law.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .hyperelastic import HyperelasticState


@dataclass(frozen=True)
class IncompressibleBiaxialState:
    stretches: np.ndarray
    nominal_stress: np.ndarray
    cauchy_stress: np.ndarray
    pressure_multiplier: np.ndarray


@dataclass(frozen=True)
class HGOMaterial:
    mu: float
    bulk: float
    k1: float
    k2: float
    theta: float
    minimum_jacobian: float = 1e-12

    def __post_init__(self):
        v=[self.mu,self.bulk,self.k1,self.k2,self.theta,self.minimum_jacobian]
        if not np.isfinite(v).all() or min(self.mu,self.bulk,self.minimum_jacobian)<=0 or min(self.k1,self.k2)<0 or not 0<=self.theta<=np.pi/2:
            raise ValueError('Require mu/bulk/minimumJ>0, k1/k2>=0 and theta in[0,pi/2], all finite')

    @property
    def directions(self):
        a=np.array([np.cos(self.theta),np.sin(self.theta),0.]);a/=np.linalg.norm(a)
        return (a,a*np.array([1.,-1.,1.]))

    def evaluate(self,deformation_gradient):
        return self._evaluate(deformation_gradient,include_bulk=True)

    def evaluate_isochoric(self,deformation_gradient):
        """Same full-tensor state API, with no volumetric energy or derivatives."""
        return self._evaluate(deformation_gradient,include_bulk=False)

    def _evaluate(self,deformation_gradient,include_bulk):
        F=np.array(deformation_gradient,dtype=float,copy=True)
        if F.shape[-2:]!=(3,3) or not np.isfinite(F).all():raise ValueError('Expected finite (...,3,3) deformation gradient')
        J=np.linalg.det(F)
        if not np.isfinite(J).all() or np.any(J<=self.minimum_jacobian):raise ValueError('Nonpositive, singular or inadmissibly small deformation Jacobian')
        H=np.swapaxes(np.linalg.inv(F),-1,-2);eye=np.eye(3);q=J**(-2/3)
        Q=np.einsum('...iL,...kJ->...iJkL',H,H);HH=np.einsum('...iJ,...kL->...iJkL',H,H)
        def invariant_derivatives(I,B,base):
            S=B-I[...,None,None]/3*H
            gradient=2*q[...,None,None]*S
            hessian=2*q[...,None,None,None,None]*(base-(2/3)*np.einsum('...iJ,...kL->...iJkL',S,H)-(2/3)*np.einsum('...iJ,...kL->...iJkL',H,B)+I[...,None,None,None,None]/3*Q)
            return gradient,hessian
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                I1=np.sum(F*F,axis=(-2,-1));G1,A1=invariant_derivatives(I1,F,np.einsum('ik,JL->iJkL',eye,eye))
                W=.5*self.mu*(q*I1-3);P=.5*self.mu*G1;A=.5*self.mu*A1
                for a in self.directions:
                    current=F@a;I4=np.sum(current*current,axis=-1)
                    # Algebraic difference of squares avoids cancellation and
                    # gives exact zero at F=I for floating-point unit vectors.
                    bar_current=J[...,None]**(-1/3)*current
                    E=np.sum((bar_current-a)*(bar_current+a),axis=-1)
                    x=np.maximum(E,0.);z=self.k2*x*x;expz=np.exp(z)
                    exprel=np.divide(np.expm1(z),z,out=np.ones_like(z),where=z!=0)
                    W=W+.5*self.k1*x*x*exprel
                    first=self.k1*x*expz;second=np.where(E>0,self.k1*expz*(1+2*z),0.)
                    B=np.einsum('...i,J->...iJ',current,a);base=np.einsum('ik,J,L->iJkL',eye,a,a)
                    G4,A4=invariant_derivatives(I4,B,base)
                    P=P+first[...,None,None]*G4
                    A=A+first[...,None,None,None,None]*A4+second[...,None,None,None,None]*np.einsum('...iJ,...kL->...iJkL',G4,G4)
                if include_bulk:
                    logJ=np.log(J);W=W+.5*self.bulk*logJ**2;P=P+self.bulk*logJ[...,None,None]*H
                    A=A+self.bulk*HH-self.bulk*logJ[...,None,None,None,None]*Q
                S=np.swapaxes(H,-1,-2)@P;sigma=P@np.swapaxes(F,-1,-2)/J[...,None,None];green=.5*(np.swapaxes(F,-1,-2)@F-eye)
        except FloatingPointError as exc:raise ValueError('HGO constitutive overflow or invalid arithmetic') from exc
        if not all(np.isfinite(x).all() for x in (W,P,A,S,sigma,green)):raise ValueError('Nonfinite HGO state')
        return HyperelasticState(F,J,W,P,S,sigma,green,A)

    def incompressible_biaxial_state(self,inplane_stretches):
        """Exact J=1, diagonal F, zero thickness nominal traction.

        Pressure multiplier uses P=Piso-p*F^{-T}; it is tied to this isochoric
        energy convention, not a separately measured physiological pressure.
        Input shape(...,2), ordered circumferential/longitudinal.
        """
        inplane=np.asarray(inplane_stretches,dtype=float)
        if inplane.shape[-1:]!=(2,) or not np.isfinite(inplane).all() or np.any(inplane<=0):raise ValueError('Expected positive finite (...,2) in-plane stretches')
        d=np.concatenate([inplane,(1/np.prod(inplane,axis=-1))[...,None]],axis=-1)
        s=self.evaluate_isochoric(np.eye(3)*d[..., :,None]);p=s.first_piola[...,2,2]*d[...,2]
        nominal=s.first_piola-p[...,None,None]*np.eye(3)/d[...,None,:]
        return IncompressibleBiaxialState(d,nominal,nominal*d[...,None,:],p)
