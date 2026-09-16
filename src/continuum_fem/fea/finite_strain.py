"""Opt-in total-Lagrangian C3D6 pilot with dead loads and displacement BCs.

Uses three triangle × two axial quadrature points, matching the retained wedge
node/quadrature order. P=dW/dF and its full material tangent are assembled in the
reference frame. No existing linear solver is changed. This displacement-only
pilot is NOT locking-free near incompressibility and has no follower pressure,
contact, viscosity, inertia, adaptive continuation or global injectivity proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from .hyperelastic import HyperelasticMaterial


@dataclass(frozen=True)
class AssemblyState:
    energy: float
    internal_force: np.ndarray
    tangent: csr_matrix | None
    quadrature: dict[str, np.ndarray]


@dataclass(frozen=True)
class NonlinearResult:
    converged: bool
    failure_reason: str | None
    load_factor: float
    displacement: np.ndarray
    applied_load: np.ndarray
    residual: np.ndarray
    reactions: np.ndarray
    state: AssemblyState
    history: tuple[dict, ...]
    accepted_steps: tuple[dict, ...]


class C3D6HyperelasticSystem:
    """Reference mesh plus one material or an explicit material per element."""
    def __init__(self, nodes, elements, materials):
        self.nodes=np.array(nodes,dtype=np.float64,copy=True)
        connectivity=np.asarray(elements)
        if self.nodes.ndim!=2 or self.nodes.shape[1]!=3 or not np.isfinite(self.nodes).all():
            raise ValueError('Expected finite [N,3] reference nodes')
        if connectivity.ndim!=2 or connectivity.shape[1]!=6 or not len(connectivity) or not np.issubdtype(connectivity.dtype,np.integer):
            raise ValueError('Expected nonempty integer [E,6] C3D6 connectivity')
        self.elements=connectivity.astype(np.int64,copy=True)
        if np.any(self.elements<0) or np.any(self.elements>=len(self.nodes)) or any(len(set(row))!=6 for row in self.elements):
            raise ValueError('Invalid or repeated C3D6 node index')
        self.materials=tuple([materials]*len(self.elements) if isinstance(materials,HyperelasticMaterial) else materials)
        if len(self.materials)!=len(self.elements) or not all(isinstance(m,HyperelasticMaterial) for m in self.materials):
            raise ValueError('Provide one HyperelasticMaterial or one per element')
        natural=[]
        for r,s in ((1/6,1/6),(2/3,1/6),(1/6,2/3)):
            for z in (-1/np.sqrt(3),1/np.sqrt(3)):
                triangle=np.array([1-r-s,r,s]);dt=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
                d=np.empty((6,3))
                for layer,sign in enumerate((-1.,1.)):
                    d[3*layer:3*layer+3,:2]=.5*(1+sign*z)*dt
                    d[3*layer:3*layer+3,2]=.5*sign*triangle
                natural.append(d)
        natural=np.asarray(natural)
        jacobian=np.einsum('eai,qaJ->eqiJ',self.nodes[self.elements],natural)
        determinant=np.linalg.det(jacobian)
        if not np.isfinite(determinant).all() or np.any(determinant<=0):
            raise ValueError('Nonpositive reference wedge Jacobian at quadrature')
        self.gradients=np.einsum('qaJ,eqJi->eqai',natural,np.linalg.inv(jacobian))
        self.weights=determinant/6
        self.reference_energy_scale=float(sum(self.weights[e].sum()*(m.mu+m.bulk+sum(f.ksi for f in m.fibers)) for e,m in enumerate(self.materials)))
        self.element_dofs=(3*self.elements[:,:,None]+np.arange(3)).reshape(len(self.elements),18)
        self.nodes.setflags(write=False);self.elements.setflags(write=False)
        self.gradients.setflags(write=False);self.weights.setflags(write=False)

    def assemble(self, displacement, *, tangent=True):
        u=np.asarray(displacement,dtype=np.float64)
        if u.shape!=self.nodes.shape or not np.isfinite(u).all():
            raise ValueError('Displacement must be finite and match reference nodes')
        F=np.eye(3)+np.einsum('eai,eqaJ->eqiJ',u[self.elements],self.gradients)
        states=[material.evaluate(F[e]) for e,material in enumerate(self.materials)]
        fields={name:np.stack([getattr(state,name) for state in states]) for name in
                ('deformation_gradient','jacobian','energy_density','first_piola','second_piola','cauchy','green_lagrange')}
        element_force=np.einsum('eq,eqiJ,eqaJ->eai',self.weights,fields['first_piola'],self.gradients)
        internal=np.zeros_like(self.nodes)
        np.add.at(internal,self.elements.ravel(),element_force.reshape(-1,3))
        stiffness=None
        if tangent:
            A=np.stack([state.tangent_dP_dF for state in states])
            element_K=np.einsum('eq,eqaJ,eqiJkL,eqbL->eaibk',self.weights,self.gradients,A,self.gradients).reshape(-1,18,18)
            row=np.broadcast_to(self.element_dofs[:,:,None],element_K.shape).ravel()
            col=np.broadcast_to(self.element_dofs[:,None,:],element_K.shape).ravel()
            stiffness=coo_matrix((element_K.ravel(),(row,col)),shape=(self.nodes.size,self.nodes.size)).tocsr()
        return AssemblyState(float(np.sum(self.weights*fields['energy_density'])),internal,stiffness,fields)

    def solve(self, loads, fixed, *, prescribed=None, steps=8, max_iterations=30,
              relative_tolerance=1e-8, absolute_tolerance=1e-10, max_backtracks=20):
        """Proportional quasi-static ramp; each accepted step is actual equilibrium.

        Loads are fixed world/reference-basis nodal dead forces, not pressure.
        Prescribed final displacements ramp proportionally at fixed DOFs only.
        Newton line search decreases total potential and rejects invalid J.
        Numerical failure returns converged=False plus its actual reached state;
        it is never silently relabelled as the requested final equilibrium.
        Reactions are internal-minus-applied forces at constrained DOFs. Energy
        is integral W dV0; 0.5*f.u is generally NOT its nonlinear work identity.
        """
        load=np.array(loads,dtype=np.float64,copy=True)
        mask=np.asarray(fixed)
        if load.shape!=self.nodes.shape or not np.isfinite(load).all() or mask.shape!=load.shape or mask.dtype!=bool:
            raise ValueError('Finite loads and boolean fixed mask must match [N,3]')
        target=np.zeros_like(load) if prescribed is None else np.asarray(prescribed,dtype=np.float64)
        if target.shape!=load.shape or not np.isfinite(target).all() or np.any(target[~mask]!=0):
            raise ValueError('Prescribed values must be finite, shaped [N,3], and zero on free DOFs')
        if not isinstance(steps,int) or steps<1 or not isinstance(max_iterations,int) or max_iterations<0 or not isinstance(max_backtracks,int) or max_backtracks<0:
            raise ValueError('Invalid Newton/load-step iteration counts')
        if not np.isfinite([relative_tolerance,absolute_tolerance]).all() or min(relative_tolerance,absolute_tolerance)<=0:
            raise ValueError('Convergence tolerances must be positive finite')
        free=~mask.ravel();u=np.zeros_like(load);history=[];accepted=[]
        factor=0.
        def result(failure):
            state=self.assemble(u,tangent=False)
            residual=state.internal_force-factor*load
            return NonlinearResult(failure is None,failure,float(factor),u.copy(),factor*load,residual,
                                   np.where(mask,residual,0.),state,tuple(history),tuple(accepted))
        initial=self.assemble(u,tangent=False)
        accepted.append({'load_factor':0.,'displacement':u.copy(),'energy':initial.energy,
                         'applied_load':np.zeros_like(load),'reactions':np.zeros_like(load)})
        for step in range(1,steps+1):
            factor=step/steps;previous=u.copy();u[mask]=factor*target[mask]
            try:state=self.assemble(u)
            except ValueError:
                u=previous;factor=(step-1)/steps
                return result('Prescribed ramp left the admissible Jacobian/material domain')
            for iteration in range(max_iterations+1):
                residual=(state.internal_force-factor*load).ravel()[free]
                scale=max(float(np.linalg.norm((factor*load).ravel()[free])),float(np.linalg.norm(state.internal_force.ravel()[free])))
                norm=float(np.linalg.norm(residual));threshold=absolute_tolerance+relative_tolerance*scale
                potential=state.energy-float(np.sum(factor*load*u))
                entry={'step':step,'iteration':iteration,'load_factor':factor,'free_residual_norm':norm,
                       'residual_scale':scale,'threshold':threshold,'potential':potential,
                       'minimum_J':float(state.quadrature['jacobian'].min())}
                history.append(entry)
                if norm<=threshold:
                    residual_full=state.internal_force-factor*load
                    accepted.append({'load_factor':factor,'displacement':u.copy(),'energy':state.energy,
                                     'applied_load':factor*load,'reactions':np.where(mask,residual_full,0.)})
                    break
                if iteration==max_iterations:return result('Maximum Newton iterations reached')
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',MatrixRankWarning)
                        direction=spsolve(state.tangent[free][:,free],-residual)
                except (MatrixRankWarning,RuntimeError,ValueError):return result('Singular or invalid constrained tangent')
                descent=float(residual@direction)
                if not np.isfinite(direction).all() or not np.isfinite(descent) or descent>=0:
                    return result('Newton direction is nonfinite or not a potential descent direction')
                alpha=1.;trial=None
                for backtrack in range(max_backtracks+1):
                    candidate=u.copy();candidate.ravel()[free]+=alpha*direction
                    try:
                        trial=self.assemble(candidate,tangent=False)
                        trial_potential=trial.energy-float(np.sum(factor*load*candidate))
                        if trial_potential<=potential+1e-4*alpha*descent:break
                        # Energy subtraction loses digits near equilibrium. Only
                        # accept a roundoff-level increase when the force test
                        # already meets its unchanged convergence threshold.
                        roundoff=128*np.finfo(float).eps*max(abs(potential),self.reference_energy_scale)
                        trial_residual=(trial.internal_force-factor*load).ravel()[free]
                        if abs(trial_potential-potential)<=roundoff and np.linalg.norm(trial_residual)<=threshold:
                            entry['roundoff_force_converged_acceptance']=True
                            break
                    except ValueError:pass
                    trial=None;alpha*=.5
                if trial is None:return result('Line search exhausted without admissible potential decrease')
                entry['accepted_alpha']=alpha;entry['backtracks']=backtrack
                u=candidate;state=self.assemble(u)
        return result(None)
