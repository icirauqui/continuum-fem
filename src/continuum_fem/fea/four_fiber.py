"""Four-fiber finite-strain material with explicit published conventions.

Ferruzzi et al. (2011), doi:10.1098/rsif.2010.0299, Eq2.1:
  W=c/2*(I1-3)+sum_i k1_i/(4*k2_i)*expm1(k2_i*(I4_i-1)^2).
Puertolas et al. (2020), doi:10.1016/j.jmbbm.2019.103507, printed Eq5:
  W=c*(I1-3)+sum_i k1_i/(2*k2_i)*expm1(k2_i*(I4_i-1)^2).
These energies differ by two for identical numerical coefficient values.
Normalization is never inferred from the data. Historical HyperFit convention
for published colon coefficients is unresolved; neither option is tissue GT.

All stress-like inputs are Pa. Reference x=circumferential, y=axial, z=thickness.
Fibers are axial, circumferential, and TWO separate equivalent +/-theta
families, with theta measured from circumferential (radians). Each diagonal
family receives diagonal_k1. Literal source response is untruncated; optional
`tension_only_diagnostic` changes the compression law and has a tangent kink.

Finite-J use replaces invariants by their isochoric counterparts and optionally
adds bulk/2*ln(J)^2: a declared numerical extension of source incompressibility.
evaluate_isochoric supplies the bulk-free state for a mixed formulation.
No viscoelasticity, contact, failure, global stability or empirical validity is
implied by a finite returned material state.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .hyperelastic import HyperelasticState
from .hgo import IncompressibleBiaxialState


@dataclass(frozen=True)
class FourFiberMaterial:
    matrix_coefficient: float
    bulk: float
    axial_k1: float
    axial_k2: float
    circumferential_k1: float
    circumferential_k2: float
    diagonal_k1: float
    diagonal_k2: float
    theta: float
    normalization: str = 'ferruzzi2011'
    activation: str = 'untruncated'
    minimum_jacobian: float = 1e-12

    def __post_init__(self):
        values=[self.matrix_coefficient,self.bulk,self.axial_k1,self.axial_k2,
                self.circumferential_k1,self.circumferential_k2,self.diagonal_k1,
                self.diagonal_k2,self.theta,self.minimum_jacobian]
        if (not np.isfinite(values).all() or min(self.matrix_coefficient,self.bulk,self.minimum_jacobian)<=0
                or min(values[2:8])<0 or not 0<=self.theta<=np.pi/2):
            raise ValueError('Require finite positive matrix/bulk/minimumJ, nonnegative fiber coefficients, theta in[0,pi/2]')
        if self.normalization not in ('ferruzzi2011','puertolas2020_literal'):
            raise ValueError('Specify ferruzzi2011 or puertolas2020_literal normalization')
        if self.activation not in ('untruncated','tension_only_diagnostic'):
            raise ValueError('Specify untruncated or tension_only_diagnostic activation')

    @property
    def energy_prefactor(self):
        return .5 if self.normalization=='ferruzzi2011' else 1.

    @property
    def mu(self):
        """Ground-matrix shear modulus, not total anisotropic tangent modulus."""
        return 2*self.energy_prefactor*self.matrix_coefficient

    @property
    def convention_metadata(self):
        return {'normalization':self.normalization,'energy_prefactor_relative_to_puertolas_printed':self.energy_prefactor,
                'activation':self.activation,'coefficient_units':'Pa; k2 dimensionless; theta radians',
                'reference_axes':['circumferential','axial','thickness'],
                'diagonal_coefficient_applies_to':'each of two equivalent diagonal families',
                'source_doi':'10.1098/rsif.2010.0299' if self.normalization=='ferruzzi2011' else '10.1016/j.jmbbm.2019.103507',
                'finite_J_extension':'isochoric invariants; optional bulk/2*log(J)^2',
                'historical_colon_coefficient_convention':'unresolved; fixed-convention training evidence is not tissue validation'}

    @property
    def families(self):
        a=np.array([np.cos(self.theta),np.sin(self.theta),0.]);a/=np.linalg.norm(a)
        return ((np.array([0.,1.,0.]),self.axial_k1,self.axial_k2),
                (np.array([1.,0.,0.]),self.circumferential_k1,self.circumferential_k2),
                (a,self.diagonal_k1,self.diagonal_k2),
                (a*np.array([1.,-1.,1.]),self.diagonal_k1,self.diagonal_k2))

    def evaluate(self,deformation_gradient):
        return self._evaluate(deformation_gradient,True)

    def evaluate_isochoric(self,deformation_gradient):
        return self._evaluate(deformation_gradient,False)

    def _evaluate(self,deformation_gradient,include_bulk):
        F=np.array(deformation_gradient,dtype=float,copy=True)
        if F.shape[-2:]!=(3,3) or not np.isfinite(F).all():raise ValueError('Expected finite (...,3,3) F')
        J=np.linalg.det(F)
        if not np.isfinite(J).all() or np.any(J<=self.minimum_jacobian):raise ValueError('Inadmissible deformation Jacobian')
        H=np.swapaxes(np.linalg.inv(F),-1,-2);eye=np.eye(3);q=J**(-2/3)
        Q=np.einsum('...iL,...kJ->...iJkL',H,H);HH=np.einsum('...iJ,...kL->...iJkL',H,H)
        def derivatives(I,B,base):
            S=B-I[...,None,None]/3*H
            gradient=2*q[...,None,None]*S
            hessian=2*q[...,None,None,None,None]*(base-(2/3)*np.einsum('...iJ,...kL->...iJkL',S,H)
                     -(2/3)*np.einsum('...iJ,...kL->...iJkL',H,B)+I[...,None,None,None,None]/3*Q)
            return gradient,hessian
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                I1=np.sum(F*F,axis=(-2,-1));G1,A1=derivatives(I1,F,np.einsum('ik,JL->iJkL',eye,eye))
                W=.5*self.mu*(q*I1-3);P=.5*self.mu*G1;A=.5*self.mu*A1
                for a,k1,k2 in self.families:
                    if k1==0:continue
                    current=F@a;I4=np.sum(current*current,axis=-1)
                    bar_current=J[...,None]**(-1/3)*current
                    E=np.sum((bar_current-a)*(bar_current+a),axis=-1)
                    x=np.maximum(E,0.) if self.activation=='tension_only_diagnostic' else E
                    z=k2*x*x;expz=np.exp(z)
                    exprel=np.divide(np.expm1(z),z,out=np.ones_like(z),where=z!=0)
                    coeff=self.energy_prefactor*k1
                    W=W+.5*coeff*x*x*exprel
                    first=coeff*x*expz;second=coeff*expz*(1+2*z)
                    if self.activation=='tension_only_diagnostic':second=np.where(E>0,second,0.)
                    G4,A4=derivatives(I4,np.einsum('...i,J->...iJ',current,a),np.einsum('ik,J,L->iJkL',eye,a,a))
                    P=P+first[...,None,None]*G4
                    A=A+first[...,None,None,None,None]*A4+second[...,None,None,None,None]*np.einsum('...iJ,...kL->...iJkL',G4,G4)
                if include_bulk:
                    logJ=np.log(J);W=W+.5*self.bulk*logJ**2;P=P+self.bulk*logJ[...,None,None]*H
                    A=A+self.bulk*HH-self.bulk*logJ[...,None,None,None,None]*Q
                S=np.swapaxes(H,-1,-2)@P;sigma=P@np.swapaxes(F,-1,-2)/J[...,None,None]
                green=.5*(np.swapaxes(F,-1,-2)@F-eye)
        except FloatingPointError as exc:raise ValueError('Four-fiber constitutive overflow or invalid arithmetic') from exc
        if not all(np.isfinite(v).all() for v in (W,P,A,S,sigma,green)):raise ValueError('Nonfinite four-fiber material state')
        return HyperelasticState(F,J,W,P,S,sigma,green,A)

    def incompressible_biaxial_state(self,inplane_stretches):
        """Exact J1 diagonal response, zero thickness traction, circ/axial input.

        P=Piso-p*F^-T; p is this isochoric formulation's constraint multiplier,
        not observed physiological pressure. The bulk penalty is not evaluated.
        """
        inplane=np.asarray(inplane_stretches,dtype=float)
        if inplane.shape[-1:]!=(2,) or not np.isfinite(inplane).all() or np.any(inplane<=0):raise ValueError('Expected positive finite (...,2) stretches')
        d=np.concatenate([inplane,(1/np.prod(inplane,axis=-1))[...,None]],axis=-1)
        state=self.evaluate_isochoric(np.eye(3)*d[..., :,None]);p=state.first_piola[...,2,2]*d[...,2]
        nominal=state.first_piola-p[...,None,None]*np.eye(3)/d[...,None,:]
        return IncompressibleBiaxialState(d,nominal,nominal*d[...,None,:],p)
