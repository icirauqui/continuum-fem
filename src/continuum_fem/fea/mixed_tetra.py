"""EXPERIMENTAL mixed TetP2/P1 finite-strain dead-load equilibrium.

Opt-in: does not alter the default linear solver or constitutive APIs. Material
must expose evaluate_isochoric(F) -> energy_density, first_piola, tangent_dP_dF.
The finite bulk log-J extension and constant consistent augmentation are owned
here. p is hydrostatic Kirchhoff TENSION, not positive-compression pressure.
Exact discrete projection p=K M^-1 integral(N logJ) is not pointwise p=K logJ.

All numeric inputs use declared solver units; explicit scales map them to SI.
Numerical convergence is not continuum, incompressibility or tissue validation.
See docs/mixed_tetra.md for equations, failures and validated scope.
"""
from dataclasses import dataclass
import warnings
import copy
import numpy as np
from scipy.sparse import coo_matrix,bmat,diags
from scipy.sparse.linalg import splu,spsolve,MatrixRankWarning
try:
    from skfem import Basis,ElementVector,ElementTetP2,ElementTetP1,LinearForm,BilinearForm,asm
    from skfem.helpers import ddot,sym_grad,div
except ImportError as exc:
    raise ImportError('Optional mixed FEM requires scikit-fem==12.0.2; see requirements-mixed.txt') from exc


@dataclass(frozen=True)
class MixedUnitScales:
    """One solver length/stress unit in metres/Pa; no automatic material rescaling."""
    length_meter: float
    stress_pascal: float

    def __post_init__(self):
        if (isinstance(self.length_meter,(bool,np.bool_)) or isinstance(self.stress_pascal,(bool,np.bool_))
            or not np.isfinite([self.length_meter,self.stress_pascal]).all() or min(self.length_meter,self.stress_pascal)<=0):
            raise ValueError('Explicit finite positive length and stress scales required')
        with np.errstate(over='ignore',invalid='ignore'):
            derived=np.array([np.float64(self.stress_pascal)*np.float64(self.length_meter)**2,
                              np.float64(self.stress_pascal)*np.float64(self.length_meter)**3])
        if not np.isfinite(derived).all() or np.any(derived<=0):raise ValueError('Physical unit conversion overflow or underflow')

    @property
    def force_newton(self):return self.stress_pascal*self.length_meter**2
    @property
    def energy_joule(self):return self.stress_pascal*self.length_meter**3
    def to_dict(self):
        return dict(length_meter=self.length_meter,stress_pascal=self.stress_pascal,
                    force_newton=self.force_newton,energy_joule=self.energy_joule)


@dataclass(frozen=True)
class MixedSolveSettings:
    steps: int = 4
    max_iterations: int = 50
    regularized: bool = True
    absolute_residual_tolerance: float = 1e-10
    relative_residual_tolerance: float = 1e-8

    def __post_init__(self):
        if (not isinstance(self.steps,int) or isinstance(self.steps,bool) or self.steps<1
            or not isinstance(self.max_iterations,int) or isinstance(self.max_iterations,bool) or self.max_iterations<0
            or not isinstance(self.regularized,bool)
            or not np.isfinite([self.absolute_residual_tolerance,self.relative_residual_tolerance]).all()
            or min(self.absolute_residual_tolerance,self.relative_residual_tolerance)<0):
            raise ValueError('Positive integer steps, nonnegative iterations/tolerances required')


class MixedTetP2P1:
    """Fixed mesh/quadrature, constant bulk and augmentation; caller supplies scales.

    mesh coordinates, isochoric material coefficients, bulk/shear, displacement,
    pressure and loads must all use the same solver-unit convention. For numeric
    SI inputs set scales=(1m,1Pa); for nondimensional inputs divide physical
    quantities consistently. No material-specific coefficient inference occurs.
    """
    experimental = True

    def __init__(self,mesh,material,*,bulk_modulus,reference_shear_modulus,
                 scales,quadrature_order=6,augmentation=0.,reference_frame=None,
                 minimum_jacobian=1e-12):
        if (not isinstance(scales,MixedUnitScales) or not callable(getattr(material,'evaluate_isochoric',None))
            or not np.isfinite([bulk_modulus,reference_shear_modulus,augmentation,minimum_jacobian]).all()
            or min(bulk_modulus,reference_shear_modulus,minimum_jacobian)<=0
            or not 0<=augmentation<bulk_modulus
            or not isinstance(quadrature_order,int) or isinstance(quadrature_order,bool) or quadrature_order<1):
            raise ValueError('Require isochoric material, explicit scales, positive moduli/J/order, and0<=augmentation<bulk')
        if mesh.p.shape[0]!=3 or mesh.t.shape[0]!=4 or not mesh.nelements or not np.isfinite(mesh.p).all():
            raise ValueError('Finite nonempty three-dimensional tetrahedral mesh required')
        for attr,value in [('bulk',bulk_modulus),('mu',reference_shear_modulus)]:
            if hasattr(material,attr) and not np.isclose(getattr(material,attr),value,rtol=1e-12,atol=0):
                raise ValueError(f'Material {attr} must already match solver stress units; scales convert exports only')
        self.material=material;self.bulk=float(bulk_modulus);self.mu=float(reference_shear_modulus)
        self.rho=float(augmentation);self.pressure_equation_scale=1-self.rho/self.bulk
        self.scales=scales;self.minimum_jacobian=float(minimum_jacobian);self.quadrature_order=quadrature_order
        self.bu=Basis(mesh,ElementVector(ElementTetP2()),intorder=quadrature_order)
        self.bp=Basis(mesh,ElementTetP1(),intorder=quadrature_order)
        if (not np.isfinite(self.bu.dx).all() or not np.all(self.bu.dx>0)
            or not np.array_equal(self.bu.X,self.bp.X) or not np.array_equal(self.bu.W,self.bp.W)
            or not np.array_equal(self.bu.dx,self.bp.dx)):
            raise ValueError('Positive, identical displacement/pressure quadrature required')
        shape=self.bu.dx.shape+(3,3)
        Q=np.eye(3) if reference_frame is None else np.asarray(reference_frame,dtype=float)
        try:Q=np.array(np.broadcast_to(Q,shape),copy=True)
        except ValueError as exc:raise ValueError('Material frame must broadcast over element/quadrature points') from exc
        if (not np.isfinite(Q).all() or not np.allclose(Q.swapaxes(-1,-2)@Q,np.eye(3),rtol=0,atol=1e-10)
            or not np.allclose(np.linalg.det(Q),1.,rtol=0,atol=1e-10)):
            raise ValueError('Material reference axes must be proper orthonormal columns')
        Q.flags.writeable=False;self.reference_frame=Q
        self._contractions=_ElementContractions(self)
        @BilinearForm
        def mass(p,q,w):return p*q
        self.pressure_mass=asm(mass,self.bp).tocsc();self._pressure_factor=splu(self.pressure_mass)

    @property
    def displacement_dofs(self):return self.bu.N
    @property
    def pressure_dofs(self):return self.bp.N

    def _check_state(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(self.bu.N+self.bp.N,) or not np.isfinite(x).all():raise ValueError('Finite complete mixed state required')
        return x

    def constitutive_state(self,F,p):
        F=np.asarray(F,dtype=float);p=np.asarray(p,dtype=float)
        if F.shape!=self.bu.dx.shape+(3,3) or not np.isfinite(F).all():raise ValueError('Finite element/quadrature deformation gradients required')
        J=np.linalg.det(F)
        if not np.isfinite(J).all() or np.any(J<=self.minimum_jacobian):raise ValueError('Inadmissible trial Jacobian')
        if p.shape!=J.shape or not np.isfinite(p).all():raise ValueError('Finite quadrature pressure required')
        Q=self.reference_frame;iso=self.material.evaluate_isochoric(F@Q)
        W=np.asarray(iso.energy_density);Pi=np.asarray(iso.first_piola);Ai=np.asarray(iso.tangent_dP_dF)
        if (W.shape!=J.shape or Pi.shape!=F.shape or Ai.shape!=J.shape+(3,3,3,3)
            or not all(np.isfinite(v).all() for v in (W,Pi,Ai))):raise ValueError('Invalid isochoric material response')
        H=np.swapaxes(np.linalg.inv(F),-1,-2);L=np.log(J);K=self.bulk;rho=self.rho;a=self.pressure_equation_scale
        Pi=Pi@np.swapaxes(Q,-1,-2);Ai=np.einsum('...iakb,...Ja,...Lb->...iJkL',Ai,Q,Q)
        cross=np.einsum('...iL,...kJ->...iJkL',H,H);outer=np.einsum('...iJ,...kL->...iJkL',H,H)
        c=L-p/K;P=Pi+p[...,None,None]*H;A=Ai-p[...,None,None,None,None]*cross
        potential=W+p*L-.5*p*p/K;internal=W+.5*p*p/K
        if rho:
            P=P+rho*c[...,None,None]*H;A=A+rho*outer-rho*c[...,None,None,None,None]*cross
            potential=potential+.5*rho*c*c;internal=internal+.5*rho*L*L-.5*(rho/K)*p**2/K
        if not all(np.isfinite(v).all() for v in (P,A,potential,internal)):
            raise ValueError('Nonfinite mixed material response')
        return dict(F=F,J=J,p=p,P=P,A=A,H=H,coupling_H=a*H,constraint=a*c,
                    unscaled_constraint=c,pressure_block=-a/K,mixed_potential_density=potential,
                    mixed_internal_energy_density=internal,pointwise_displacement_energy_density=W+.5*K*L**2,
                    effective_kirchhoff_pressure=p+rho*c,cauchy=P@np.swapaxes(F,-1,-2)/J[...,None,None],
                    pointwise_pressure_defect=p-K*L)

    def assemble(self,x,tangent=True):return self._contractions.assemble(self._check_state(x),tangent)

    def project_pressure(self,x):
        x=self._check_state(x).copy();ug=self.bu.interpolate(x[:self.bu.N]).grad
        J=np.linalg.det(np.moveaxis(ug,(0,1),(-2,-1))+np.eye(3))
        if not np.isfinite(J).all() or np.any(J<=self.minimum_jacobian):raise ValueError('Inadmissible trial Jacobian')
        L=np.log(J)
        @LinearForm
        def rhs(q,w):return L*q
        x[self.bu.N:]=self.bulk*self._pressure_factor.solve(asm(rhs,self.bp))
        if not np.isfinite(x).all():raise ValueError('Nonfinite projected pressure')
        return x

    def reduced_evaluate(self,x,force,tangent=True):
        force=np.asarray(force,dtype=float)
        if force.shape!=(self.bu.N,) or not np.isfinite(force).all():raise ValueError('Finite displacement dead load required')
        x=self.project_pressure(x);r,T,state=self.assemble(x,tangent);r[:self.bu.N]-=force
        internal=float(np.sum(self.bu.dx*state['mixed_internal_energy_density']));work=float(force@x[:self.bu.N])
        return x,r,T,state,internal-work,abs(internal)+abs(work)

    def reference_shear_diagonal(self):
        @BilinearForm
        def shear(u,v,w):return 2*self.mu*(ddot(sym_grad(u),sym_grad(v))-div(u)*div(v)/3)
        diagonal=asm(shear,self.bu).diagonal()
        if not np.isfinite(diagonal).all() or np.any(diagonal<=0):raise ValueError('Nonpositive reference shear diagonal')
        return diagonal

    def solve(self,load,fixed,*,prescribed_displacement=None,settings=MixedSolveSettings(),progress=None,initial_state=None):
        if not isinstance(settings,MixedSolveSettings):raise ValueError('MixedSolveSettings required')
        if progress is not None and not callable(progress):raise ValueError('Progress observer must be callable')
        nu=self.bu.N;size=nu+self.bp.N;load=np.asarray(load,dtype=float);fixed=np.asarray(fixed)
        target=np.zeros(nu) if prescribed_displacement is None else np.asarray(prescribed_displacement,dtype=float)
        if (load.shape!=(nu,) or not np.isfinite(load).all() or fixed.ndim!=1
            or not np.issubdtype(fixed.dtype,np.integer) or np.any(fixed<0) or np.any(fixed>=nu)
            or len(np.unique(fixed))!=len(fixed) or target.shape!=(nu,) or not np.isfinite(target).all()):
            raise ValueError('Finite dead load/prescribed displacement and unique displacement-only fixed DOFs required')
        free=np.setdiff1d(np.arange(size),fixed);free_u=free[free<nu];diagonal=self.reference_shear_diagonal() if settings.regularized else None
        x=np.zeros(size) if initial_state is None else self._check_state(initial_state).copy();history=[];accepted=[];state=None;current_force=np.zeros(nu);factor=0.
        def emit(event):
            if progress is None:return None
            try:
                if progress(copy.deepcopy(event)) is False:return 'cancelled by progress callback'
            except Exception as exc:return f'progress callback failed: {type(exc).__name__}: {exc}'
            return None
        def norm(r):
            scaled=r.copy();scaled[nu:]*=self.bulk/self.pressure_equation_scale
            return float(np.linalg.norm(scaled[free]))
        def finish(reason):
            residual=None;energies=None
            if state is not None:
                residual,_,st=self.assemble(x,False);residual[:nu]-=current_force
                internal=float(np.sum(self.bu.dx*st['mixed_internal_energy_density']));work=float(current_force@x[:nu])
                energies=dict(internal=internal,force_displacement_pairing=work,reduced_potential=internal-work,
                              internal_joule=internal*self.scales.energy_joule,force_displacement_pairing_joule=work*self.scales.energy_joule)
            return dict(converged=reason is None,failure=reason,x=x.copy(),state=state,history=history,
                        accepted=accepted,load_factor=factor,load=current_force.copy(),fixed_displacement_dofs=fixed.copy(),
                        residual=residual,reactions=None if residual is None else residual[fixed].copy(),
                        reactions_newton=None if residual is None else residual[fixed]*self.scales.force_newton,
                        energies=energies,units=self.scales.to_dict(),experimental=True,
                        initialization='zero' if initial_state is None else 'supplied predictor; not implicitly accepted',
                        qualification='discrete stationarity only; stability/volume/locking/mesh/tissue accuracy gates remain separate')
        for step in range(1,settings.steps+1):
            factor=step/settings.steps;current_force=factor*load;x[fixed]=factor*target[fixed];state=None
            for iteration in range(settings.max_iterations+1):
                state=None
                try:x,r,T,state,energy,energy_scale=self.reduced_evaluate(x,current_force)
                except (ValueError,FloatingPointError) as exc:return finish(f'inadmissible current state: {exc}')
                nr=norm(r);history.append(dict(step=step,iteration=iteration,scaled_free_residual=nr,
                    free_force_residual_norm=float(np.linalg.norm(r[free_u])),
                    unscaled_constraint_residual_norm=float(np.linalg.norm(r[nu:])/self.pressure_equation_scale),
                    reduced_potential=energy,minimum_J=float(state['J'].min())))
                stopped=emit(dict(event='iteration',load_factor=factor,x=x,iteration=history[-1],units=self.scales.to_dict()))
                if stopped:return finish(stopped)
                threshold=settings.absolute_residual_tolerance+settings.relative_residual_tolerance*np.linalg.norm(current_force[free_u])
                if nr<threshold:break
                if iteration==settings.max_iterations:return finish('maximum Newton iterations')
                dx,slope,attempts=regularized_direction(T,r,free,nu,diagonal);history[-1]['direction_attempts']=attempts
                if dx is None:return finish('no finite descent reduced Newton direction')
                history[-1]['directional_energy_derivative']=slope;alpha=1.
                for _ in range(20):
                    trial=x.copy();trial[free_u]+=alpha*dx[free_u]
                    try:
                        trial,rr,_,_,trial_energy,trial_scale=self.reduced_evaluate(trial,current_force,False)
                        exact=trial_energy<=energy+1e-4*alpha*slope
                        tolerance=64*np.finfo(float).eps*max(energy_scale,trial_scale,1e-12)
                        roundoff=(abs(alpha*slope)<=tolerance and trial_energy<=energy+tolerance and norm(rr)<(1-1e-4*alpha)*nr)
                        stationary=(alpha==1. and attempts[-1]['shift']==0. and norm(rr)<threshold)
                        if exact or roundoff or stationary:break
                    except (ValueError,FloatingPointError):pass
                    alpha*=.5
                else:return finish('reduced energy line search exhausted')
                x=trial;history[-1]['alpha']=alpha;history[-1]['roundoff_residual_safeguard']=bool(not exact and roundoff)
                history[-1]['strict_stationary_trial_exception']=bool(stationary and not exact and not roundoff)
            accepted.append(dict(load_factor=factor,x=x.copy(),reduced_potential=energy,
                                 scaled_free_residual=nr,minimum_J=float(state['J'].min())))
            stopped=emit(dict(event='accepted',state=accepted[-1],units=self.scales.to_dict()))
            if stopped:return finish(stopped)
        x,r,_,state,_,_=self.reduced_evaluate(x,load,False)
        return finish(None)

    def physical_fields(self,x,*,include_fixed_pressure_tangent=False):
        """Material/kinematic evaluation; does not assert input equilibrium."""
        x=self._check_state(x);_,_,s=self.assemble(x,False);L=self.scales.length_meter;S=self.scales.stress_pascal
        if not isinstance(include_fixed_pressure_tangent,bool):raise ValueError('Tangent export flag must be boolean')
        fields=dict(displacement_dof_reference_meter=self.bu.doflocs.T*L,
                    quadrature_reference_position_meter=np.asarray(self.bu.global_coordinates()).transpose(1,2,0)*L,
                    reference_material_frame=self.reference_frame.copy(),
                    green_lagrange_strain=.5*(s['F'].swapaxes(-1,-2)@s['F']-np.eye(3)),
                    second_piola_pascal=np.linalg.solve(s['F'],s['P'])*S,
                    effective_pressure_defect_pascal=(s['effective_kirchhoff_pressure']-self.bulk*np.log(s['J']))*S,
                    displacement_coefficients_meter=x[:self.bu.N]*L,pressure_coefficients_pascal=x[self.bu.N:]*S,
                    deformation_gradient=s['F'],jacobian=s['J'],kirchhoff_pressure_pascal=s['p']*S,
                    effective_kirchhoff_pressure_pascal=s['effective_kirchhoff_pressure']*S,
                    pressure_defect_pascal=s['pointwise_pressure_defect']*S,first_piola_pascal=s['P']*S,
                    cauchy_pascal=s['cauchy']*S,quadrature_reference_volume_meter3=self.bu.dx*L**3,
                    mixed_internal_energy_joule=float(np.sum(self.bu.dx*s['mixed_internal_energy_density']))*self.scales.energy_joule,
                    units=self.scales.to_dict(),equilibrium_verified=False)
        if include_fixed_pressure_tangent:
            fields['fixed_pressure_mixed_tangent_dP_dF_pascal']=s['A']*S
        return fields


def regularized_direction(T,r,free,nu,reference_diagonal=None):
    """Equivalent reduced-Hessian direction; shifts only affect trial direction."""
    free_u=free[free<nu];shifts=[0.] if reference_diagonal is None else [0.,1e-4,1e-2,1.,100.,10000.]
    base=T[free][:,free];diagonal=np.zeros(len(r))
    if reference_diagonal is not None:diagonal[:nu]=reference_diagonal
    attempts=[]
    for shift in shifts:
        matrix=base if shift==0 else base+diags(shift*diagonal[free]);dx=np.zeros(len(r))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning);dx[free]=spsolve(matrix,-r[free])
            slope=float(r[free_u]@dx[free_u]);finite=bool(np.isfinite(dx).all() and np.isfinite(slope))
        except (RuntimeError,MatrixRankWarning):finite=False;slope=np.nan
        attempts.append(dict(shift=shift,finite=finite,slope=slope if finite else None))
        if finite and slope<0:return dx,slope,attempts
    return None,None,attempts


class _ElementContractions:
    def __init__(self,system):
        self.system=system
        scalar=Basis(system.bu.mesh,ElementTetP2(),quadrature=(system.bu.X,system.bu.W))
        components=system.bu.split_indices()
        for indices in components:
            np.testing.assert_array_equal(system.bu.doflocs[:,indices],scalar.doflocs)
        # Physical-reference gradients, not parametric element gradients.
        self.G=np.stack([field[0].grad for field in scalar.basis],axis=-1).transpose(1,2,3,0)
        self.N=np.stack([np.asarray(field[0]) for field in system.bp.basis],axis=-1)
        self.udofs=np.stack([indices[scalar.element_dofs.T] for indices in components],axis=-1)
        self.pdofs=system.bp.element_dofs.T
        self.flat_u=self.udofs.reshape(len(self.udofs),-1)
        self.dx=system.bu.dx
        np.testing.assert_array_equal(system.bu.dx,system.bp.dx)

    @staticmethod
    def matrix(values,rows,columns,shape):
        rr=np.broadcast_to(rows[:,:,None],values.shape).ravel()
        cc=np.broadcast_to(columns[:,None,:],values.shape).ravel()
        return coo_matrix((values.ravel(),(rr,cc)),shape=shape).tocsr()

    def assemble(self,x,tangent=True):
        system=self.system;nu=system.bu.N;np_=system.bp.N
        ug=np.einsum('eai,eqaj->eqij',x[self.udofs],self.G,optimize=True)
        F=ug+np.eye(3);p=np.einsum('ea,eqa->eq',x[nu+self.pdofs],self.N,optimize=True)
        st=system.constitutive_state(F,p)
        ulocal=np.einsum('eq,eqij,eqaj->eai',self.dx,st['P'],self.G,optimize=True)
        plocal=np.einsum('eq,eq,eqa->ea',self.dx,st['constraint'],self.N,optimize=True)
        r=np.r_[np.bincount(self.udofs.ravel(),weights=ulocal.ravel(),minlength=nu),
                np.bincount(self.pdofs.ravel(),weights=plocal.ravel(),minlength=np_)]
        T=None
        if tangent:
            # Two-stage contraction avoids repeated full constitutive contractions
            # for every vector basis pair. No mass lumping or reduced integration.
            AG=np.einsum('eqijkl,eqbl->eqijkb',st['A'],self.G,optimize=True)
            uu=np.einsum('eq,eqaj,eqijkb->eaibk',self.dx,self.G,AG,optimize=True).reshape(len(self.udofs),30,30)
            H=st.get('coupling_H',st['H'])
            up=np.einsum('eq,eqij,eqaj,eqb->eaib',self.dx,H,self.G,self.N,optimize=True).reshape(len(self.udofs),30,4)
            pp=np.einsum('eq,eqa,eqb->eab',self.dx,self.N,self.N,optimize=True)*st['pressure_block']
            U=self.matrix(uu,self.flat_u,self.flat_u,(nu,nu))
            B=self.matrix(up,self.flat_u,self.pdofs,(nu,np_))
            C=self.matrix(pp,self.pdofs,self.pdofs,(np_,np_))
            T=bmat([[U,B],[B.T,C]],format='csr')
        return r,T,st
