#!/usr/bin/env python3
import argparse, pathlib, numpy as np
import firedrake as fd
from firedrake_adjoint import *     # Control, ReducedFunctional, tape utils
import icepack

# ---------------------------- Physics blocks ---------------------------------
def friction_stress(u, C, m):
    return -C * fd.sqrt(fd.inner(u, u)) ** (1.0 / m - 1.0) * u

def friction(**kwargs):
    u = kwargs["velocity"]
    theta = kwargs["log_friction"]
    m     = kwargs.get("sliding_exponent", fd.Constant(1.0))
    C0    = kwargs["friction"]
    grounded = kwargs.get("grounded", None)
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
    A  = A0 * fd.exp(phi)
    return icepack.models.viscosity.viscosity_depth_averaged(
        velocity=u, thickness=h, fluidity=A
    )

# ---------------------------- Flotation & masks -------------------------------
def flotation_height_from_bed(bed, rho_i, rho_w):
    hf = (rho_w/rho_i) * fd.max_value(-bed, 0.0)   # flotation thickness
    s_f = bed + hf                                  # flotation surface height
    return s_f

def grounded_mask(surface, s_float, Q):
    z_above = fd.interpolate(surface - s_float, Q)
    floating = fd.interpolate(fd.conditional(z_above < 0, 1.0, 0.0), Q)
    grounded = fd.interpolate(1.0 - floating, Q)
    return floating, grounded

# ---------------------------- I/O ---------------------------------------------
def load_from_checkpoint(path, gamma):
    with fd.CheckpointFile(str(path), "r") as chk:
        mesh = chk.load_mesh(name="flask")
        h0   = chk.load_function(mesh, name="thickness")
        s0   = chk.load_function(mesh, name="surface")
        b    = chk.load_function(mesh, name="bed")
        uobs = chk.load_function(mesh, name="velocity_obs")
        sig  = chk.load_function(mesh, name="sigma_obs")
        C0   = chk.load_function(mesh, name="friction")
        A0   = chk.load_function(mesh, name="fluidity")
        theta_map = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_friction")
        phi_map   = chk.load_function(mesh, name=f"gamma{gamma:.2e}_log_fluidity")
        u   = chk.load_function(mesh, name=f"gamma{gamma:.2e}_velocity")
    V = uobs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h0, s0, b, uobs, sig, C0, A0, theta_map, phi_map, u

# ---------------------------- Solver ------------------------------------------
def build_solver(mesh):
    model = icepack.models.IceStream(friction=friction, viscosity=viscosity)
    side_ids = list(mesh.exterior_facets.unique_markers)
    opts = {
        "side_wall_ids": side_ids,
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 80,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "prognostic_solver_parameters": {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    return icepack.solvers.FlowSolver(model, **opts)

# --------------------- QoI: Volume Above Flotation ----------------------------
def qoi_vaf_form(h, s, b, mesh, rho_i, rho_w):
    s_float = flotation_height_from_bed(b, rho_i, rho_w)
    haf = fd.max_value(s - s_float, 0.0)     # height above flotation
    return haf * fd.dx(mesh)

# --------------------------- Forward loop -------------------------------------
def run_forward(mesh, V, Q, h0, s0, b, C0, A0, theta_map, phi_map, u,
                T_years, nsteps, n_sens=1,
                out_ckpt="../mesh/flask.h5", outdir="../results/forward",
                accum_rate=0.0, accum_control=False):

    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    dt = fd.Constant(T_years/float(nsteps), domain=mesh)

    alpha = fd.Constant(1.0, domain=mesh)
    accum = fd.Function(Q, name="accum")

    h = h0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)

    rho_i = icepack.constants.ice_density
    rho_w = icepack.constants.water_density

    s_float0 = flotation_height_from_bed(b, rho_i, rho_w)
    floating, grounded = grounded_mask(s, s_float0, Q)

    solver = build_solver(mesh)

    sens_idxs = [nsteps] if n_sens <= 1 else np.linspace(1, nsteps, n_sens, dtype=int).tolist()
    results = []

    get_working_tape().clear_tape()
    continue_annotation()

    ctrl_theta = Control(theta_map)
    ctrl_phi   = Control(phi_map)
    ctrl_alpha = Control(alpha) if accum_control else None

    with fd.CheckpointFile(str(out_ckpt), "a") as chk:
        for k in range(1, nsteps+1):
            accum.assign(alpha * float(accum_rate))

            u = solver.diagnostic_solve(
                velocity=u, thickness=h, surface=s,
                log_friction=theta_map, friction=C0,
                log_fluidity=phi_map,   fluidity=A0,
                grounded=grounded,
                sliding_exponent=fd.Constant(1.0, domain=mesh)
            )
            h = solver.prognostic_solve(thickness=h, velocity=u, accumulation=accum, dt=dt)
            s.interpolate(b + h)

            if k in sens_idxs:
                J_form = qoi_vaf_form(h, s, b, mesh, rho_i, rho_w)
                J = fd.assemble(J_form)
                VAFk = float(J)

                rf_theta = ReducedFunctional(J, ctrl_theta)
                dJ_dtheta = rf_theta.derivative()
                gθ = fd.Function(Q, name=f"dVAF_dtheta_t{k:04d}"); gθ.interpolate(dJ_dtheta)

                rf_phi = ReducedFunctional(J, ctrl_phi)
                dJ_dphi = rf_phi.derivative()
                gφ = fd.Function(Q, name=f"dVAF_dphi_t{k:04d}"); gφ.interpolate(dJ_dphi)

                if accum_control:
                    rf_alpha = ReducedFunctional(J, ctrl_alpha)
                    dJ_dalpha = rf_alpha.derivative()
                    αfun = fd.Function(Q, name=f"dVAF_dalpha_t{k:04d}")
                    αfun.assign(float(dJ_dalpha))

                chk.save_function(h, name="thickness_timeseries", idx=k)
                chk.save_function(u, name="velocity_timeseries",  idx=k)
                chk.save_function(gθ, name="dVAF_dtheta_timeseries", idx=k)
                chk.save_function(gφ, name="dVAF_dphi_timeseries",   idx=k)
                if accum_control:
                    chk.save_function(αfun, name="dVAF_dalpha_timeseries", idx=k)

                results.append((k, VAFk))

    ts = outdir / "vaf_timeseries.csv"
    with open(ts, "w") as f:
        f.write("step,VAF\n")
        for k, VAFk in results:
            f.write(f"{k},{VAFk}\n")
    print(f"[OK] Wrote VAF time series to {ts}")
    return results

# ------------------------------ CLI -----------------------------------------

ckpt="../mesh/flask.h5"
gamma = 3.0
T = 20.0
n = 80
s = 5
outdir="../results"

mesh, V, Q, h0, s0, b, uobs, sig, C0, A0, theta_map, phi_map, u = load_from_checkpoint(ckpt, gamma)
a = firedrake.Constant(0.0,domain=mesh)

run_forward(mesh, V, Q, h0, s0, b, C0, A0, theta_map, phi_map, uobs,
            T_years=T, nsteps=n, n_sens=s,
            out_ckpt=ckpt, outdir=outdir,
            accum_rate=a,
            accum_control=False)

