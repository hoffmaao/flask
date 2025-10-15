#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Fig. 10 (a, e, f) using ARPACK eigenpairs (no SLEPc).

Inputs expected:
- ../mesh/venable.h5  (mesh + MAP + time series sensitivities written by run_forward_venable)
- ../results/eigenvalues.txt  (eigenvalues from ARPACK K^{-1}H_GN eigendec)
- eigenmodes 'mode_0000', 'mode_0001', ... stored either in ../mesh/venable.h5 or ../mesh/venable_modes.h5

Outputs:
- ../results/fig10a_sd_alpha_patch1km.png (patch-avg SD map of α=C0*exp(θ))
- ../results/fig10e_vaf_bands.png        (VAF(t) with prior & posterior 2σ bands)
- ../results/fig10f_sigma_convergence.png (posterior σ(t_end) vs number of modes)
"""

import pathlib, math, numpy as np
import matplotlib.pyplot as plt
import firedrake as fd

# ----------------------------- Paths & options --------------------------------
CKPT              = "../mesh/venable.h5"
MODES_FILE        = "../mesh/venable_modes.h5"   # if modes were saved separately; else set to CKPT
USE_MODES_FILE    = pathlib.Path(MODES_FILE).exists()
EIG_TXT           = "../results/eigenvalues.txt"
VAF_CSV           = "../results/forward/vaf_timeseries.csv"

# number of eigenpairs to use in uncertainty reductions (you can trim)
NEV_MAX           = 800            # keep modest (hundreds) for speed/robustness
PATCH_RADIUS_M    = 1000.0         # 1-km disks for panel (a)
PATCH_STRIDE      = 3              # compute SD every 3rd P1 node (set 1 for full-res)
SMOOTH_SD_M       = 3000.0         # final cosmetic smoothing length for SD map
FIG_DPI           = 250

# Prior hyperparameters (must match inversion settings)
GAMMA_THETA = 3.0;  ELL_THETA = 7.5e3; THETA_SCALE = 1.0
GAMMA_PHI   = 1.0;  ELL_PHI   = 7.5e3; PHI_SCALE   = 1.0

# ----------------------------- I/O helpers -----------------------------------
def load_base(ckpt, gamma):
    with fd.CheckpointFile(str(ckpt), "r") as chk:
        mesh  = chk.load_mesh(name="venable")
        h     = chk.load_function(mesh, name="thickness")
        s     = chk.load_function(mesh, name="surface")
        b     = chk.load_function(mesh, name="bed")
        uobs  = chk.load_function(mesh, name="velocity_obs")
        sigma = chk.load_function(mesh, name="sigma_obs")
        C0    = chk.load_function(mesh, name="friction")
        A0    = chk.load_function(mesh, name="fluidity")
        thMAP = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        phMAP = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
    V = uobs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h, s, b, uobs, sigma, C0, A0, thMAP, phMAP

def load_eigs(eig_txt, cap=None):
    vals = []
    for line in pathlib.Path(eig_txt).read_text().splitlines():
        i, lam = line.strip().split()
        vals.append(float(lam))
    if cap is not None:
        vals = vals[:cap]
    return np.asarray(vals, dtype=float)

def load_modes(file_path, mesh, Q, nev):
    QQ = fd.MixedFunctionSpace([Q, Q])
    modes = []
    with fd.CheckpointFile(str(file_path), "r") as chk:
        for i in range(nev):
            name = f"mode_{i:04d}"
            try:
                z = chk.load_function(mesh, name=name)
            except Exception:
                break
            # ensure space/layout
            if z.function_space().num_sub_spaces() == 2:
                modes.append(z.copy(deepcopy=True))
            else:
                # stored as single field? convert if needed (unlikely)
                raise RuntimeError(f"{name} is not a MixedFunction")
    return modes

# ----------------------------- Physics helpers --------------------------------
def flotation_surface_from_bed(bed, rho_i, rho_w):
    return bed + (rho_w/rho_i) * fd.max_value(-bed, 0.0)

def grounded_mask(surface, s_float, Q):
    # light smoothing to avoid speckle
    q0 = fd.interpolate(surface - s_float, Q)
    q  = q0.copy(deepcopy=True)
    J = 0.5*((q-q0)**2 + (100.0)**2 * fd.inner(fd.grad(q), fd.grad(q))) * fd.dx
    F = fd.derivative(J, q); fd.solve(F == 0, q)
    floating = fd.interpolate(fd.conditional(q < 0, 1.0, 0.0), Q)
    grounded = fd.interpolate(1.0 - floating, Q)
    return floating, grounded

# ----------------------------- Prior solvers K^{-1} ---------------------------
def make_Kinv(mesh, Q, invA, gamma, ell, scale):
    y = fd.Function(Q); z = fd.Function(Q)
    v, w = fd.TrialFunction(Q), fd.TestFunction(Q)
    a = ( invA*fd.inner(v, w) + gamma*(ell/scale)**2 * invA * fd.inner(fd.grad(v), fd.grad(w)) ) * fd.dx(mesh)
    L = invA * fd.inner(z, w) * fd.dx(mesh)
    pr = fd.LinearVariationalProblem(a, L, y)
    sp = {"ksp_type":"cg","ksp_rtol":1e-10,"pc_type":"hypre","pc_hypre_type":"boomeramg"}
    so = fd.LinearVariationalSolver(pr, solver_parameters=sp)
    def apply(rhs):
        z.assign(rhs); so.solve(); return y.copy(deepcopy=True)
    return apply

def K_energy(mesh, invA, gamma_th, ell_th, s_th, gamma_ph, ell_ph, s_ph, vth, vph):
    return float(fd.assemble(
        invA*((vth*vth) + (vph*vph))
        + gamma_th*(ell_th/s_th)**2 * invA * fd.inner(fd.grad(vth), fd.grad(vth))
        + gamma_ph*(ell_ph/s_ph)**2 * invA * fd.inner(fd.grad(vph), fd.grad(vph))
        * fd.dx(mesh)
    ))

# ----------------------------- Panel (a): SD map ------------------------------
def sd_map_alpha_disks(mesh, Q, C0, theta_map, eigvals, modes, Kθ_inv,
                       gamma_theta, ell_theta, theta_scale, invA,
                       R=1000.0, stride=3, smooth_len=3000.0):
    # K-normalize modes (combined θ,φ energy)
    γθ = fd.Constant(gamma_theta, domain=mesh)
    ℓθ = fd.Constant(ell_theta,   domain=mesh)
    sθ = fd.Constant(theta_scale, domain=mesh)
    γφ = fd.Constant(1.0, domain=mesh)   # φ part present but unused for θ-only SD
    ℓφ = fd.Constant(1.0, domain=mesh)
    sφ = fd.Constant(1.0, domain=mesh)

    modes_kn = []
    for z in modes:
        e = K_energy(mesh, invA, γθ, ℓθ, sθ, γφ, ℓφ, sφ, z.sub(0), z.sub(1))
        if e <= 0: continue
        sc = 1.0/math.sqrt(e)
        zi = fd.Function(z.function_space())
        zi.sub(0).assign(sc * z.sub(0)); zi.sub(1).assign(sc * z.sub(1))
        modes_kn.append(zi)
    modes = modes_kn

    alpha_map = fd.Function(Q); alpha_map.interpolate(C0*fd.exp(theta_map))
    sd_alpha  = fd.Function(Q, name="sd_alpha_patch1km"); sd_alpha.assign(0.0)

    # cheap, robust "disk" at node j: start with nodal delta; diffuse to ~R; normalize by area
    chi = fd.Function(Q); q = fd.Function(Q)
    v, w = fd.TrialFunction(Q), fd.TestFunction(Q)

    dofs = np.arange(Q.dof_dset.size)[::max(1, stride)]
    for j in dofs:
        chi.assign(0.0); chi.dat.data[j] = 1.0
        # diffuse once with length ~R
        J = 0.5*((chi-q)**2 + (R**2) * fd.inner(fd.grad(chi), fd.grad(chi))) * fd.dx
        F = fd.derivative(J, chi); fd.solve(F == 0, chi)
        Aj = float(fd.assemble(chi * fd.dx(mesh)))
        if Aj <= 0.0:
            continue
        gpatch = fd.Function(Q); gpatch.assign(chi / Aj)

        # prior variance: g^T K^{-1} g
        y = Kθ_inv(gpatch)
        prior_var = float(fd.assemble(gpatch * y * fd.dx(mesh)))

        # reduction: sum_i <K^{-1}g, v_i^θ>^2 * λ/(1+λ)
        proj = np.array([float(fd.assemble(y * z.sub(0) * fd.dx(mesh))) for z in modes])
        red  = float(np.sum( (proj**2) * (eigvals[:len(proj)]/(1.0 + eigvals[:len(proj)])) ))
        post_var = max(prior_var - red, 0.0)

        sd_alpha.dat.data[j] = float(alpha_map.dat.data_ro[j]) * math.sqrt(post_var)

    # cosmetic smoothing for continuity
    q0 = sd_alpha.copy(deepcopy=True)
    J = 0.5*((sd_alpha - q0)**2 + (smooth_len**2) * fd.inner(fd.grad(sd_alpha), fd.grad(sd_alpha))) * fd.dx
    F = fd.derivative(J, sd_alpha); fd.solve(F == 0, sd_alpha)
    return sd_alpha

# ----------------------------- Panel (e,f): bands -----------------------------
def load_sens_timeseries(ckpt, mesh, Q):
    """Return dict k -> (gθ_k, gφ_k). Stops when an idx is missing."""
    out = {}
    with fd.CheckpointFile(str(ckpt), "r") as chk:
        k = 1
        while True:
            try:
                gθ = chk.load_function(mesh, name="dVAF_dtheta_timeseries", idx=k)
                gφ = chk.load_function(mesh, name="dVAF_dphi_timeseries",   idx=k)
            except Exception:
                break
            out[k] = (gθ.copy(deepcopy=True), gφ.copy(deepcopy=True))
            k += 1
    return out

def variance_time_series(mesh, invA, eigvals, modes, Kθ_inv, Kφ_inv, sens_dict):
    # K-normalize modes for stability (combined θ,φ energy)
    γθ = fd.Constant(GAMMA_THETA, domain=mesh)
    ℓθ = fd.Constant(ELL_THETA,   domain=mesh)
    sθ = fd.Constant(THETA_SCALE, domain=mesh)
    γφ = fd.Constant(GAMMA_PHI,   domain=mesh)
    ℓφ = fd.Constant(ELL_PHI,     domain=mesh)
    sφ = fd.Constant(PHI_SCALE,   domain=mesh)
    modes_kn = []
    for z in modes:
        e = K_energy(mesh, invA, γθ, ℓθ, sθ, γφ, ℓφ, sφ, z.sub(0), z.sub(1))
        if e <= 0: continue
        sc = 1.0/math.sqrt(e)
        zi = fd.Function(z.function_space())
        zi.sub(0).assign(sc*z.sub(0)); zi.sub(1).assign(sc*z.sub(1))
        modes_kn.append(zi)
    modes = modes_kn

    steps = sorted(sens_dict.keys())
    prior  = np.zeros(len(steps))
    post   = np.zeros(len(steps))
    # also gather per-mode reductions at the last time for panel (f)
    reductions_last = None

    for j,k in enumerate(steps):
        gθ, gφ = sens_dict[k]
        yθ, yφ = Kθ_inv(gθ), Kφ_inv(gφ)
        prior[j] = float(fd.assemble(gθ*yθ * fd.dx(mesh) + gφ*yφ * fd.dx(mesh)))

        # coefficients: <K^{-1}g, v_i> = ∫ (yθ v_i^θ + yφ v_i^φ) dx
        proj = np.array([float(fd.assemble(yθ*z.sub(0) * fd.dx(mesh) + yφ*z.sub(1) * fd.dx(mesh)))
                         for z in modes])
        red_by_mode = (proj**2) * (eigvals[:len(proj)]/(1.0 + eigvals[:len(proj)]))
        post[j] = max(prior[j] - float(np.sum(red_by_mode)), 0.0)

        if j == len(steps)-1:
            reductions_last = np.cumsum(red_by_mode)

    return steps, prior, post, reductions_last

# ----------------------------------- Main ------------------------------------
# 1) Load base fields & constants
GAMMA = 3.0
mesh, V, Q, h, s, b, uobs, sigma_vec, C0, A0, theta_map, phi_map = load_base(CKPT, GAMMA)  # matches your forward/inv I/O
rho_i = fd.Constant(icepack.constants.ice_density, domain=mesh)
rho_w = fd.Constant(icepack.constants.water_density, domain=mesh)
area  = float(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)))
invA  = fd.Constant(1.0/area, domain=mesh)

# 2) Read eigenvalues/modes (ARPACK products)
eigvals_all = load_eigs(EIG_TXT)
NEV = min(NEV_MAX, len(eigvals_all))
eigvals = eigvals_all[:NEV]
modes   = load_modes(MODES_FILE if USE_MODES_FILE else CKPT, mesh, Q, NEV)

# 3) Build K^{-1} solvers for both controls (identical to eigendec/forward)
Kθ_inv = make_Kinv(mesh, Q, invA, fd.Constant(GAMMA_THETA,domain=mesh),
                   fd.Constant(ELL_THETA,domain=mesh), fd.Constant(THETA_SCALE,domain=mesh))
Kφ_inv = make_Kinv(mesh, Q, invA, fd.Constant(GAMMA_PHI,domain=mesh),
                   fd.Constant(ELL_PHI,domain=mesh), fd.Constant(PHI_SCALE,domain=mesh))

# 4) Panel (a): SD map of α using 1-km patches
sdα = sd_map_alpha_disks(mesh, Q, C0, theta_map, eigvals, modes, Kθ_inv,
                         GAMMA_THETA, ELL_THETA, THETA_SCALE, invA,
                         R=PATCH_RADIUS_M, stride=PATCH_STRIDE, smooth_len=SMOOTH_SD_M)
fig, ax = plt.subplots(figsize=(4.6,4.0))
m = fd.tripcolor(sdα, axes=ax, cmap="magma")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(m, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cb.set_label("SD(α)  [SI units]")
fig.tight_layout()
out = pathlib.Path("../figures/fig10a_sd_alpha_patch1km.png")
fig.savefig(out, dpi=FIG_DPI); plt.close(fig)
print(f"[OK] wrote {out}")

# 5) Panels (e,f): VAF(t) bands & σ convergence (last time)
sens = load_sens_timeseries(CKPT, mesh, Q)
steps, prior_var, post_var, red_last_cumsum = variance_time_series(
    mesh, invA, eigvals, modes, Kθ_inv, Kφ_inv, sens
)
t = np.array(steps, dtype=float)   # assume unit time steps; scale if desired
prior_2s = 2.0*np.sqrt(prior_var)
post_2s  = 2.0*np.sqrt(post_var)

# If you saved the mean VAF(t) CSV in your forward (optional plot)
vaf_mean = None
csvp = pathlib.Path(VAF_CSV)
if csvp.exists():
    vals = []
    for line in csvp.read_text().splitlines()[1:]:
        k,v = line.split(",")
        vals.append(float(v))
    vaf_mean = np.array(vals, dtype=float)

# Panel (e)
fig, ax = plt.subplots(figsize=(6.2,3.4))
if vaf_mean is not None and len(vaf_mean)>=len(t):
    ax.plot(t, vaf_mean[:len(t)], lw=1.5, label="VAF (mean)")
ax.fill_between(t,  +(prior_2s/2.0),  -(prior_2s/2.0), alpha=0.15, label="prior ±1σ", step="mid")
ax.fill_between(t,  +(post_2s/2.0),   -(post_2s/2.0),  alpha=0.30, label="posterior ±1σ", step="mid")
ax.set_xlabel("time index (arbitrary)"); ax.set_ylabel("ΔVAF sensitivity band")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
out = pathlib.Path("../figures/fig10e_vaf_bands.png")
fig.savefig(out, dpi=FIG_DPI); plt.close(fig)
print(f"[OK] wrote {out}")

# Panel (f): σ convergence vs number of modes (last time)
if red_last_cumsum is not None:
    prior_last = prior_var[-1]
    m = np.arange(1, len(red_last_cumsum)+1)
    sigma_post_m = np.sqrt(np.maximum(prior_last - red_last_cumsum, 0.0))
    fig, ax = plt.subplots(figsize=(4.6,3.4))
    ax.plot(m, sigma_post_m, lw=1.8)
    ax.set_xlabel("number of modes m"); ax.set_ylabel("posterior σ (last time)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = pathlib.Path("../figures/fig10f_sigma_convergence.png")
    fig.savefig(out, dpi=FIG_DPI); plt.close(fig)
    print(f"[OK] wrote {out}")
