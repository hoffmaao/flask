#!/usr/bin/env python3
import icepack
import argparse, pathlib, csv
import numpy as np
import firedrake as fd

# ----------------- utilities -----------------
def l2_inner(mesh, f, g):
    """L2 inner product ∫ f*g dx. Assumes scalar fields on same space or compatible."""
    return float(fd.assemble(fd.inner(f, g) * fd.dx(mesh)))

def try_load_function(chk, mesh, name, idx=None):
    """Load a function by name, with optional time index; raise clean error if missing."""
    try:
        if idx is None:
            return chk.load_function(mesh, name=name)
        else:
            return chk.load_function(mesh, name=name, idx=idx)
    except Exception as e:
        raise RuntimeError(f"Could not load '{name}' (idx={idx}): {e}")

def split_mode_to_Q(mode, Q):
    """
    Return a pair (z_theta, z_phi) of scalar Functions on space Q.
    If 'mode' is scalar: return (mode_on_Q, None).
    If 'mode' is mixed:  return (mode.sub(0) projected to Q, mode.sub(1) projected to Q).
    """
    fs = mode.function_space()
    try:
        nsub = fs.num_sub_spaces()
    except Exception:
        nsub = 1

    if nsub == 1:
        if mode.function_space() == Q:
            return mode, None
        else:
            return fd.project(mode, Q), None
    else:
        zt = fd.project(mode.sub(0), Q)
        zp = fd.project(mode.sub(1), Q)
        return zt, zp

def load_eigenvalues(path):
    """Read eigenvalues from a text file with either 'i val' per line or just 'val'."""
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 1:
                vals.append(float(parts[0]))
            else:
                vals.append(float(parts[1]))
    return np.asarray(vals)

# ----------------- core computation -----------------
def analytic_mode_contributions_dual(mesh, g_theta, g_phi, lambdas, modes_Q):
    """
    Contributions of each eigenmode to QoI variance at one time.
    For a mixed mode z_i = (zθ_i, zφ_i), use a_i = <gθ, zθ_i> + <gφ, zφ_i>, then contrib = a_i^2 / λ_i.
    For a scalar mode, use a_i = <gθ, z_i> (friction‑only legacy).
    """
    contribs = []
    for lam, (zth, zph) in zip(lambdas, modes_Q):
        if zph is None:
            a = l2_inner(mesh, g_theta, zth)
        else:
            a = l2_inner(mesh, g_theta, zth) + l2_inner(mesh, g_phi, zph)
        contribs.append((a*a) / lam)
    return contribs

def detect_qoi_and_grad_names(chk, mesh):
    """
    Prefer VAF sensitivity names if present; otherwise fall back to generic 'dQ_*'.
    Returns (grad_theta_name, grad_phi_name, qoi_type) where qoi_type is 'vaf' or 'generic'.
    """
    # test availability at idx=1 (any valid index will do just for name detection)
    try:
        _ = chk.load_function(mesh, "dVAF_dtheta_timeseries", idx=1)
        _ = chk.load_function(mesh, "dVAF_dphi_timeseries",   idx=1)
        return "dVAF_dtheta_timeseries", "dVAF_dphi_timeseries", "vaf"
    except Exception:
        return "dQ_dtheta_timeseries", "dQ_dphi_timeseries", "generic"

def compute_qoi_at_step(mesh, h, s, b, qoi_type):
    """Compute QoI value at the timestep for info in CSV (VAF or ∫h^2 dx)."""
    if qoi_type == "vaf":
        rho_i = icepack.constants.ice_density
        rho_w = icepack.constants.water_density
        s_float = b + (rho_w/rho_i) * fd.max_value(-b, 0.0)
        haf = fd.max_value(s - s_float, 0.0)
        return float(fd.assemble(haf * fd.dx(mesh)))
    else:
        return float(fd.assemble((h*h) * fd.dx(mesh)))

# ----------------- main -----------------
ckpt = "../mesh/flask.h5"
eigvals = "../results/eigenvalues.txt"
timesteps = [20,40,60,80]
prefex = "mode_"
nmodes = 40
outdir = "../results"




outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

# eigenvalues
lambdas = load_eigenvalues(eigvals)

with fd.CheckpointFile(str(ckpt), "r") as chk:
    # mesh and spaces
    mesh = chk.load_mesh(name="flask")
    # try to fetch a scalar Q space for projection (from friction field)
    C0 = chk.load_function(mesh, name="friction")
    Q  = C0.function_space()

    # load eigenmodes (scalar or mixed), then project components to Q
    modes_Q = []
    # count: either user sets --nmodes, or derive from eigvals, or stop on first missing
    max_modes = nmodes if nmodes is not None else len(lambdas)
    for i in range(max_modes):
        nm = f"mode_{i:04d}"
        try:
            f = chk.load_function(mesh, name=nm)
        except Exception:
            # stop if not found (allow mismatch between eigvals length and saved modes)
            break
        zth, zph = split_mode_to_Q(f, Q)
        modes_Q.append((zth, zph))

    if not modes_Q:
        raise RuntimeError(f"No eigenmodes found with prefix in {ckpt}")

    if nmodes is None:
        # align the number of eigenvalues to the number of loaded modes
        lambdas = lambdas[:len(modes_Q)]
    else:
        # enforce length
        lambdas = lambdas[:min(len(lambdas), len(modes_Q))]

    # pick gradient names (VAF or generic)
    gθ_name, gφ_name, qoi_type = detect_qoi_and_grad_names(chk, mesh)

    # Collect summary across timesteps
    summary_rows = []
    for k in timesteps:
        # Load gradients at this timestep
        gθ = try_load_function(chk, mesh, gθ_name, idx=k)
        try:
            gφ = try_load_function(chk, mesh, gφ_name, idx=k)
        except Exception:
            gφ = None  # support friction‑only cases

        # Load state to compute QoI value for reporting
        h = try_load_function(chk, mesh, "thickness_timeseries", idx=k)
        try:
            s = try_load_function(chk, mesh, "surface_timeseries", idx=k)
        except Exception:
            # if surface not saved, reconstruct as b + h
            b = try_load_function(chk, mesh, "bed")
            s = fd.Function(h.function_space()); s.interpolate(b + h)

        # (need 'b' for VAF)
        b = try_load_function(chk, mesh, "bed")

        # Project gradients to Q (safety if spaces differ)
        gθQ = gθ if gθ.function_space() == Q else fd.project(gθ, Q)
        gφQ = None if gφ is None else (gφ if gφ.function_space() == Q else fd.project(gφ, Q))

        # build per‑mode contributions
        contribs = analytic_mode_contributions_dual(
            mesh, gθQ, (gφQ if gφ is not None else fd.Function(Q, val=0.0)),
            lambdas, modes_Q
        )

        total_var = float(np.sum(contribs))
        # QoI value (VAF or ∫h^2 dx) at this step
        Qk = compute_qoi_at_step(mesh, h, s, b, qoi_type)

        # write per‑mode contributions for this timestep
        with open(outdir / f"contribs_t{k:04d}.txt", "w") as f:
            for j, (lam, c) in enumerate(zip(lambdas, contribs)):
                f.write(f"{j} {lam:.16e} {c:.16e}\n")

        # store in summary (time coordinate not known here; record step index)
        summary_rows.append((k, Qk, np.sqrt(max(total_var, 0.0))))

    # Write summary CSV
    with open(outdir / "uq_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step_index", "QoI_value", "sigma_Q"])
        for row in summary_rows:
            w.writerow(row)

print(f"[OK] Wrote per‑timestep mode files to {outdir}/contribs_t####.txt")
print(f"[OK] Wrote summary to {outdir}/uq_results.csv")

