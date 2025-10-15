"""
Eigendecomposition (two controls: theta=log-friction, phi=log-fluidity) for the FLASK domain
Adapted from the working ISMIP‑C run_eigendec pattern, extended to 2 controls.

What this script does:
  1) Loads the FLASK checkpoint written by run_inv.py (mesh/flask.h5).
  2) Builds a reduced forward (simulation, misfit, prior) for (theta, phi).
  3) Records a tlm_adjoint tape at the MAP (theta_map, phi_map) and creates a
     CachedHessian to compute Gauss–Newton Hessian actions of the *misfit only*.
  4) Builds prior solvers K_theta^{-1}, K_phi^{-1} and runs an ARPACK eigensolve
     on K^{-1}H (whitened) to compute leading eigenpairs.
  5) Saves eigenmodes into the same checkpoint as MixedFunctions and writes eigenvalues.

This mirrors the structure of the ISMIP‑C example (single control), cf. your working
ISMIP‑C script, but using two controls and FLASK names.
"""

import argparse
import pathlib
import numpy as np
import firedrake as fd
import icepack

# --- tlm_adjoint (Firedrake backend) ---
try:
    import tlm_adjoint.firedrake as tla
    from tlm_adjoint.firedrake import Functional, CachedHessian, reset_manager, start_manager, stop_manager
    HAVE_TLM = True
except Exception as exc:
    HAVE_TLM = False
    _TLM_ERR = exc

# --- SciPy for ARPACK ---
try:
    from scipy.sparse.linalg import LinearOperator, eigsh
    HAVE_SCIPY = True
except Exception as exc:
    HAVE_SCIPY = False
    _SCIPY_ERR = exc


# ---------------------------- Physics blocks ---------------------------------
def friction_stress(u, C, m):
    return -C * fd.sqrt(fd.inner(u, u)) ** (1.0 / m - 1.0) * u

def friction(**kwargs):
    """Weertman form with parameterization theta = log(C/C0)."""
    u = kwargs["velocity"]
    theta = kwargs["log_friction"]
    m = kwargs.get("sliding_exponent", fd.Constant(1.0))
    C0 = kwargs["friction"]
    grounded = kwargs["grounded"]
    C = C0 * fd.exp(theta)
    if grounded is not None:
        C = C * grounded
    tau = friction_stress(u, C, m)
    return -m / (m + 1.0) * fd.inner(tau, u)

def viscosity(**kwargs):
    A0 = kwargs['fluidity']
    u  = kwargs['velocity']
    h  = kwargs['thickness']
    phi= kwargs['log_fluidity']
    A = A0 * fd.exp(phi)
    return icepack.models.viscosity.viscosity_depth_averaged(velocity=u, thickness=h, fluidity=A)


# ---------------------------- I/O helpers ------------------------------------
def load_from_checkpoint(path, gamma):
    """Names match your run_inv.py: mesh name 'flask', and optimized fields saved as:
       gamma{γ:.2e}_log_friction, gamma{γ:.2e}_log_fluidity, etc.
    """
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh = chk.load_mesh(name="flask")
        h    = chk.load_function(mesh, name="thickness")
        s    = chk.load_function(mesh, name="surface")
        b    = chk.load_function(mesh, name="bed")
        uobs = chk.load_function(mesh, name="velocity_obs")
        sig  = chk.load_function(mesh, name="sigma_obs")
        C0   = chk.load_function(mesh, name="friction")
        A0   = chk.load_function(mesh, name="fluidity")
        u    = chk.load_function(mesh, name=f"gamma{gamma:.2e}_velocity")
        theta= chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        phi  = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
    V = uobs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h, s, b, uobs, sig, C0, A0, theta, phi


def flotation_height(bed, rho_i, rho_w):
    return fd.max_value(-bed * (rho_w/rho_i - 1.0), 0.0)

def smooth(q0, alpha=2e3):
    q = q0.copy(deepcopy=True)
    J = 0.5 * ((q - q0)**2 + alpha**2 * fd.inner(fd.grad(q), fd.grad(q))) * fd.dx
    F = fd.derivative(J, q)
    fd.solve(F == 0, q)
    return q

def grounded_mask(surface, zF, Q):
    z_above = smooth(fd.interpolate(surface - zF, Q), alpha=100.0)
    floating = fd.interpolate(fd.conditional(z_above < 0, fd.Constant(1.0), fd.Constant(0.0)), Q)
    grounded = fd.interpolate(fd.Constant(1.0) - floating, Q)
    return floating, grounded


# ------------------------- Reduced pieces (dual) ------------------------------
def build_reduced_pieces_dual(mesh, V, Q, h, s, uobs, C0, A0, grounded,
                              sigma_vec, *, gamma_theta, ell_theta, theta_scale,
                              gamma_phi,   ell_phi,   phi_scale):
    """Return simulation([theta,phi]), misfit(u), prior([theta,phi]), params."""
    # area normalizer
    area = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
    invA = fd.Constant(1.0 / area, domain=mesh)

    # Solver configured à la ISMIP‑C example: use side walls, avoid Dirichlet in tape
    model = icepack.models.IceStream(friction=friction, viscosity=viscosity)
    side_ids = list(mesh.exterior_facets.unique_markers)
    solver = icepack.solvers.FlowSolver(
        model,
        dirichlet_ids=side_ids,
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 80,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    def simulation(controls):
        theta, phi = controls
        return solver.diagnostic_solve(
            velocity=uobs, thickness=h, surface=s,
            log_friction=theta,  friction=C0,
            log_fluidity=phi,    fluidity=A0,
            grounded=grounded,
            sliding_exponent=fd.Constant(1.0, domain=mesh),
        )

    def misfit(u):
        du = u - uobs
        # componentwise σ: 0.5/area * ((dux/σx)^2 + (duy/σy)^2)
        return 0.5 * invA * ((du[0]/sigma_vec[0])**2 + (du[1]/sigma_vec[1])**2) * fd.dx(mesh)

    def prior(controls):
        theta, phi = controls
        γθ = fd.Constant(gamma_theta, domain=mesh); ℓθ = fd.Constant(ell_theta, domain=mesh); sθ = fd.Constant(theta_scale, domain=mesh)
        γφ = fd.Constant(gamma_phi,   domain=mesh); ℓφ = fd.Constant(ell_phi,   domain=mesh); sφ = fd.Constant(phi_scale,   domain=mesh)
        Jθ = 0.5 * γθ * (ℓθ/sθ)**2 * fd.inner(fd.grad(theta), fd.grad(theta)) * invA * fd.dx(mesh)
        Jφ = 0.5 * γφ * (ℓφ/sφ)**2 * fd.inner(fd.grad(phi),   fd.grad(phi))   * invA * fd.dx(mesh)
        return Jθ + Jφ

    params = {"invA": invA}
    return simulation, misfit, prior, params


# ---------------------- TLM/adjoint Hessian actions (dual) -------------------
def make_Hv_tlm_dual(simulation, misfit, theta_map, phi_map):
    """Return Hv_pair(vθ, vφ) using tlm_adjoint CachedHessian and misfit-only J."""
    if not HAVE_TLM:
        raise RuntimeError(f"tlm_adjoint not available: {_TLM_ERR}")
    reset_manager()

    def forward(theta, phi):
        th = fd.Function(theta.function_space()); th.assign(theta)
        ph = fd.Function(phi.function_space());   ph.assign(phi)
        u  = simulation([th, ph])
        J = Functional(name="J")      # <-- this is the key change
        J.assign(misfit(u))  # MISFIT ONLY for GN Hessian of data
        return J

    start_manager()
    J_map = forward(theta_map, phi_map)
    stop_manager()

    H = CachedHessian(J_map)

    def Hv_pair(vtheta, vphi):
        _, _, ddJ = H.action([theta_map, phi_map], [vtheta, vphi])
        return ddJ[0].riesz_representation("L2"), ddJ[1].riesz_representation("L2")

    return Hv_pair


# ---------------------- Prior K^{-1} (two components) ------------------------
def make_prior_solve_components(mesh, Q, invA,
                                gamma_theta, ell_theta, theta_scale,
                                gamma_phi,   ell_phi,   phi_scale,
                                use_lu=False):
    def one_solver(gamma, ell, scale):
        y = fd.Function(Q); z = fd.Function(Q)
        v, w = fd.TrialFunction(Q), fd.TestFunction(Q)
        a = ( invA * fd.inner(v, w) + gamma*(ell/scale)**2 * invA * fd.inner(fd.grad(v), fd.grad(w)) ) * fd.dx
        L = invA * fd.inner(z, w) * fd.dx
        pr = fd.LinearVariationalProblem(a, L, y)

        sp = {"ksp_type":"cg","ksp_rtol":1e-10,"pc_type":"hypre","pc_hypre_type":"boomeramg"}
        so = fd.LinearVariationalSolver(pr, solver_parameters=sp)
        def apply(rhs):
            z.assign(rhs); so.solve(); return y.copy(deepcopy=True)
        return apply
    Kθ_inv = one_solver(fd.Constant(gamma_theta, domain=mesh), fd.Constant(ell_theta, domain=mesh), fd.Constant(theta_scale, domain=mesh))
    Kφ_inv = one_solver(fd.Constant(gamma_phi,   domain=mesh), fd.Constant(ell_phi,   domain=mesh), fd.Constant(phi_scale,   domain=mesh))
    return Kθ_inv, Kφ_inv


# ---------------------- ARPACK eigensolve for K^{-1}H (dual) ----------------
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

    # sort by descending
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]

    # Return modes as MixedFunctions
    QQ = fd.MixedFunctionSpace([Q, Q])
    modes = []
    for i in range(vals.size):
        z = fd.Function(QQ)
        z.sub(0).dat.data[:] = vecs[:n, i]
        z.sub(1).dat.data[:] = vecs[n:, i]
        modes.append(z)
    return vals, modes


# ---------------------------------- CLI --------------------------------------
ckpt = "../mesh/flask.h5"
gamma = 3.0
gamma_theta =3.0
ell_theta=7.5e3
theta_scale=1.0
gamma_phi=1.0
ell_phi=7.5e3
phi_scale=1.0
k=40


# Load checkpoint and fields
mesh, V, Q, h, s, b, uobs, sigma_vec, C0, A0, theta_map, phi_map = load_from_checkpoint(ckpt, gamma)

# Grounding mask (optional but typical for shelves); multiply into C0 OR pass to action
# Here we pass to action via 'grounded' argument:
rho_i = icepack.constants.ice_density
rho_w = icepack.constants.water_density
zF = flotation_height(b, rho_i, rho_w)
floating, grounded = grounded_mask(s, zF, Q)

# Area normalization
area_val = fd.assemble(fd.Constant(1.0) * fd.dx(mesh))
invA     = fd.Constant(1.0 / area_val,domain=mesh)


# Build reduced pieces (dual controls)
simulate, misfit, prior, params = build_reduced_pieces_dual(
        mesh, V, Q, h, s, uobs, C0, A0, grounded, sigma_vec,
        gamma_theta=gamma_theta, ell_theta=ell_theta, theta_scale=theta_scale,
        gamma_phi=gamma_phi,     ell_phi=ell_phi,     phi_scale=phi_scale)

# TLM/adjoint Hessian (misfit only)
Hv_pair = make_Hv_tlm_dual(simulate, misfit, theta_map, phi_map)

# Prior K^{-1} for both components
Kθ_inv, Kφ_inv = make_prior_solve_components(
        mesh, Q, invA,
        gamma_theta=gamma_theta, ell_theta=ell_theta, theta_scale=theta_scale,
        gamma_phi=gamma_phi,     ell_phi=ell_phi,     phi_scale=phi_scale
    )

# Eigensolve for K^{-1}H
vals, modes = eigs_whitened_pair(Q, Hv_pair, Kθ_inv, Kφ_inv, k=k)

# Save modes back into the same checkpoint
with fd.CheckpointFile(ckpt, "a") as chk:
    for i, mode in enumerate(modes):
        mode.rename(f"mode_{i:04d}")
        chk.save_function(mode)

# Write eigenvalues
out = pathlib.Path(f"../results/eigenvalues.txt")
with open(out, "w") as f:
    for i, lam in enumerate(vals):
        f.write(f"{i} {lam:.16e}\n")

print(f"[OK] Saved {len(modes)} modes to {ckpt}")
print(f"[OK] Wrote eigenvalues to {out.resolve()}")

