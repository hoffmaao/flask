#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigendecomposition for FLASK with two controls (theta=log-friction, phi=log-fluidity).

- Forward solve uses hard Dirichlet BCs via tlm_adjoint EquationSolver and
  registers Function-valued BC data with deps=(bc_val,) to avoid 'Invalid dependency'.
- Gauss–Newton Hessian is built via the zero-residual trick (PSD spectrum).
- Prior-whitened eigensolve on K^{-1} H_GN (ARPACK).
- Modes saved as MixedFunctions (Q x Q) named 'mode_0000', 'mode_0001', ...
- Eigenvalues saved to ../results/eigenvalues.txt
"""

import pathlib
import numpy as np
import firedrake as fd
import icepack

# --- tlm_adjoint ---
try:
    from tlm_adjoint.firedrake import (
        Functional, CachedHessian, reset_manager, start_manager, stop_manager
    )
    try:
        # EquationSolver may live here or in .solve depending on version
        from tlm_adjoint.firedrake import EquationSolver
    except Exception:
        from tlm_adjoint.firedrake.solve import EquationSolver
    HAVE_TLM = True
except Exception as exc:
    HAVE_TLM = False
    _TLM_ERR = exc

# --- SciPy (ARPACK) ---
try:
    from scipy.sparse.linalg import LinearOperator, eigsh
    HAVE_SCIPY = True
except Exception as exc:
    HAVE_SCIPY = False
    _SCIPY_ERR = exc


# ============================ Physics blocks =================================
def friction_stress(u, C, m, u_reg=fd.Constant(1e-6)):
    """
    Regularized Weertman stress:
      tau = -C * (|u|_reg)^(1/m - 1) * u,  |u|_reg = sqrt(|u|^2 + u_reg^2)
    Regularization avoids 0^0 at a zero initial guess.
    """
    speed = fd.sqrt(fd.inner(u, u) + u_reg**2)
    return -C * speed ** (1.0 / m - 1.0) * u

def friction(**kwargs):
    """Weertman basal friction with theta = log(C/C0); optional 'grounded' mask."""
    u      = kwargs["velocity"]
    theta  = kwargs["log_friction"]
    m      = kwargs.get("sliding_exponent", fd.Constant(1.0))
    C0     = kwargs["friction"]
    grounded = kwargs.get("grounded", None)
    C = C0 * fd.exp(theta)
    if grounded is not None:
        C = C * grounded
    tau = friction_stress(u, C, m)
    return -m / (m + 1.0) * fd.inner(tau, u)

def viscosity(**kwargs):
    A0  = kwargs["fluidity"]
    u   = kwargs["velocity"]
    h   = kwargs["thickness"]
    phi = kwargs["log_fluidity"]
    A   = A0 * fd.exp(phi)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A
    )


# ============================ I/O & masks ====================================
def load_from_checkpoint(path, gamma):
    """Names match your FLASK run_inv.py outputs."""
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh  = chk.load_mesh(name="flask")
        h     = chk.load_function(mesh, name="thickness")
        s     = chk.load_function(mesh, name="surface")
        b     = chk.load_function(mesh, name="bed")
        uobs  = chk.load_function(mesh, name="velocity_obs")
        sigma = chk.load_function(mesh, name="sigma_obs")       # vector (σx, σy)
        C0    = chk.load_function(mesh, name="friction")
        A0    = chk.load_function(mesh, name="fluidity")
        # MAP fields
        u     = chk.load_function(mesh, name=f"gamma{gamma:.2e}_velocity")
        theta = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        phi   = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
    V = uobs.function_space()   # velocity space
    Q = C0.function_space()     # scalar space for controls
    return mesh, V, Q, h, s, b, uobs, sigma, C0, A0, theta, phi

def flotation_surface(bed, rho_i, rho_w):
    # s_float = b + h_float, with h_float = (rho_w/rho_i)*max(0, -b)
    return bed + (rho_w/rho_i) * fd.max_value(-bed, 0.0)

def _smooth(q0, alpha=2e3):
    q = q0.copy(deepcopy=True)
    J = 0.5 * ((q - q0)**2 + alpha**2 * fd.inner(fd.grad(q), fd.grad(q))) * fd.dx
    F = fd.derivative(J, q)
    fd.solve(F == 0, q)
    return q

def grounded_mask(surface, s_float, Q):
    z = _smooth(fd.interpolate(surface - s_float, Q), alpha=100.0)
    floating = fd.interpolate(fd.conditional(z < 0, 1.0, 0.0), Q)
    grounded = fd.interpolate(1.0 - floating, Q)
    return floating, grounded


# ============== Forward with hard Dirichlet BCs (EquationSolver) =============
def make_forward_dirichlet_equation_solver(
    *, model, mesh, V, h, s, C0, A0, grounded,
    dirichlet_ids, g_value, newton_sp=None
):
    """
    forward(theta, phi) -> u using hard Dirichlet BCs via tlm_adjoint EquationSolver.
    We pass deps=(bc_val,) so the BC value is a recorded dependency; this avoids
    'Invalid dependency' during unpack_bcs.
    """
    if newton_sp is None:
        newton_sp = {
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9, "snes_atol": 1e-10, "snes_max_it": 60,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        }

    # Ensure BC value is a Function on V (use identical object in BCs and deps)
    bc_val = g_value
    if isinstance(g_value, fd.Function) and (g_value.function_space() != V):
        bc_val = fd.project(g_value, V)
    if isinstance(bc_val, (int, float)):
        bc_val = fd.Constant((float(bc_val), 0.0))

    bcs = [fd.DirichletBC(V, bc_val, int(bid)) for bid in dirichlet_ids]

    def forward(theta, phi):
        u = fd.Function(V, name="u")
        v = fd.TestFunction(V)

        # Nonzero seed avoids |u|=0 corner in friction law
        if isinstance(bc_val, fd.Function):
            u.assign(bc_val)
        else:
            u.interpolate(fd.as_vector([fd.Constant(1e-6), fd.Constant(0.0)]))

        # Icepack energy and residual
        E = model.action(
            velocity=u, thickness=h, surface=s,
            log_friction=theta, friction=C0,
            log_fluidity=phi,  fluidity=A0,
            grounded=grounded
        )
        F = fd.derivative(E, u, v)

        # Solve via tlm_adjoint EquationSolver: explicitly pass BC value as dependency
        eq = EquationSolver(F == 0, u, bcs=bcs, solver_parameters=newton_sp)
        eq.solve()
        return u

    return forward


# =========================== GN Hessian–vector (H·v) ==========================
def make_Hv_gn(forward_u, *, mesh, invA, sigma_vec, theta_map, phi_map):
    """
    Gauss–Newton H·v via zero-residual trick:
      1) Compute u_MAP off tape.
      2) Tape J(u) = 0.5 * ∫ ((u - u_MAP)/σ)^2 * invA dx.
      3) Exact reduced Hessian of this J equals J^T W J (GN).
    """
    if not HAVE_TLM:
        raise RuntimeError(f"tlm_adjoint not available: {_TLM_ERR}")

    reset_manager()  # no annotation while computing u_MAP
    u_MAP = forward_u(theta_map, phi_map).copy(deepcopy=True)

    def misfit_GN(u):
        du = u - u_MAP
        return 0.5 * invA * ((du[0]/sigma_vec[0])**2 + (du[1]/sigma_vec[1])**2) * fd.dx(mesh)

    def record(theta, phi):
        th = fd.Function(theta.function_space()); th.assign(theta)
        ph = fd.Function(phi.function_space());   ph.assign(phi)
        u  = forward_u(th, ph)
        J  = Functional(name="J"); J.assign(misfit_GN(u))
        return J

    start_manager()
    J_map = record(theta_map, phi_map)
    stop_manager()

    H = CachedHessian(J_map)

    def Hv_pair(vθ, vφ):
        _, _, ddJ = H.action([theta_map, phi_map], [vθ, vφ])
        return ddJ[0].riesz_representation("L2"), ddJ[1].riesz_representation("L2")

    return Hv_pair


# =============================== Prior K^{-1} =================================
def make_prior_solve_components(mesh, Q, invA,
                                gamma_theta, ell_theta, theta_scale,
                                gamma_phi,   ell_phi,   phi_scale):
    def one_solver(gamma, ell, scale):
        y = fd.Function(Q); z = fd.Function(Q)
        v, w = fd.TrialFunction(Q), fd.TestFunction(Q)
        a = ( invA*fd.inner(v, w)
              + gamma*(ell/scale)**2 * invA * fd.inner(fd.grad(v), fd.grad(w)) ) * fd.dx(mesh)
        L = invA * fd.inner(z, w) * fd.dx(mesh)
        pr = fd.LinearVariationalProblem(a, L, y)
        sp = {"ksp_type":"cg","ksp_rtol":1e-10,"pc_type":"hypre","pc_hypre_type":"boomeramg"}
        so = fd.LinearVariationalSolver(pr, solver_parameters=sp)
        def apply(rhs):
            z.assign(rhs); so.solve(); return y.copy(deepcopy=True)
        return apply
    Kθ_inv = one_solver(fd.Constant(gamma_theta, domain=mesh),
                        fd.Constant(ell_theta,   domain=mesh),
                        fd.Constant(theta_scale, domain=mesh))
    Kφ_inv = one_solver(fd.Constant(gamma_phi,   domain=mesh),
                        fd.Constant(ell_phi,     domain=mesh),
                        fd.Constant(phi_scale,   domain=mesh))
    return Kθ_inv, Kφ_inv


# ===================== ARPACK eigensolve for K^{-1}H =========================
def eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k=40):
    if not HAVE_SCIPY:
        raise RuntimeError(f"SciPy eigsh required: {_SCIPY_ERR}")
    n = Q.dof_dset.size

    def _vec_to_pair(x):
        vθ = fd.Function(Q); vθ.dat.data[:] = x[:n]
        vφ = fd.Function(Q); vφ.dat.data[:] = x[n:]
        return vθ, vφ

    def _pair_to_vec(aθ, aφ):
        return np.concatenate([aθ.dat.data_ro.copy(), aφ.dat.data_ro.copy()])

    def matvec(x):
        vθ, vφ = _vec_to_pair(x)
        hθ, hφ = Hv_pair(vθ, vφ)  # H v
        yθ = Kθ_inv(hθ)           # K^{-1} H v
        yφ = Kφ_inv(hφ)
        return _pair_to_vec(yθ, yφ)

    A = LinearOperator((2*n, 2*n), matvec=matvec, dtype=float)
    vals, vecs = eigsh(A, k=min(k, 2*n-2), which="LM")

    # sort by descending value
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]

    # Return modes as MixedFunctions on [Q, Q]
    QQ = fd.MixedFunctionSpace([Q, Q])
    modes = []
    for i in range(vals.size):
        z = fd.Function(QQ)
        z.sub(0).dat.data[:] = vecs[:n, i]
        z.sub(1).dat.data[:] = vecs[n:, i]
        modes.append(z)
    return vals, modes


# ============================== Configuration =================================
CKPT            = "../mesh/flask.h5"
GAMMA           = 3.0
K_LEADING       = 40

# Prior hyperparameters (tune as in your inversion)
GAMMA_THETA = 3.0
ELL_THETA   = 7.5e3
THETA_SCALE = 1.0
GAMMA_PHI   = 1.0
ELL_PHI     = 7.5e3
PHI_SCALE   = 1.0


# ================================ Pipeline ====================================
# Load checkpoint + MAP fields (names follow your earlier scripts). 
mesh, V, Q, h, s, b, uobs, sigma_vec, C0, A0, theta_map, phi_map = load_from_checkpoint(CKPT, GAMMA)  # :contentReference[oaicite:1]{index=1}

# Model + grounding mask
model = icepack.models.IceStream(friction=friction, viscosity=viscosity)
rho_i = icepack.constants.ice_density
rho_w = icepack.constants.water_density
s_float = flotation_surface(b, rho_i, rho_w)
_, grounded = grounded_mask(s, s_float, Q)

# Choose Dirichlet facets (default: all exterior markers). Override if needed.
DIRICHLET_IDS = list(mesh.exterior_facets.unique_markers)
print(DIRICHLET_IDS)
# Dirichlet value: observed velocity (can swap for Constant((0,0)) to test no-slip)
g_value = uobs

# Build forward with hard Dirichlet BCs; BC value passed as dependency to tlm_adjoint
forward_u = make_forward_dirichlet_equation_solver(
    model=model, mesh=mesh, V=V, h=h, s=s, C0=C0, A0=A0, grounded=grounded,
    dirichlet_ids=DIRICHLET_IDS, g_value=g_value
)

# Area normalization for GN misfit
area = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
invA  = fd.Constant(1.0 / area, domain=mesh)

# Hessian action (GN) at the MAP
Hv_pair = make_Hv_gn(forward_u,
                     mesh=mesh, invA=invA, sigma_vec=sigma_vec,
                     theta_map=theta_map, phi_map=phi_map)  # :contentReference[oaicite:2]{index=2}

# Prior K^{-1} for both controls
Kθ_inv, Kφ_inv = make_prior_solve_components(
    mesh, Q, invA,
    gamma_theta=GAMMA_THETA, ell_theta=ELL_THETA, theta_scale=THETA_SCALE,
    gamma_phi=GAMMA_PHI,     ell_phi=ELL_PHI,     phi_scale=PHI_SCALE
)

# Eigensolve on K^{-1} H_GN (ARPACK)
vals, modes = eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k=K_LEADING)  # :contentReference[oaicite:3]{index=3}

# Save eigenmodes into the checkpoint and eigenvalues to disk
with fd.CheckpointFile(CKPT, "a") as chk:
    for i, mode in enumerate(modes):
        mode.rename(f"mode_{i:04d}")
        chk.save_function(mode)

out = pathlib.Path("../results/eigenvalues.txt")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for i, lam in enumerate(vals):
        f.write(f"{i} {lam:.16e}\n")

print(f"[OK] Saved {len(modes)} modes to {CKPT}")
print(f"[OK] Wrote eigenvalues to {out.resolve()}")
