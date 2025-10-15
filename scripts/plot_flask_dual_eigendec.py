#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot FLASK dual-control eigendecomposition results (ISMIP‑C style, but with 2 controls).

Features:
  • Semilogy eigenspectrum (reads a text file of eigenvalues).
  • Grid plots of leading modes for theta (log‑friction) and phi (log‑fluidity).
  • ParaView‑friendly time series for browsing modes (two series: theta_mode.pvd, phi_mode.pvd).
  • Robust loader: works with MixedFunction names like "flask_dual_mode_0000" or
    "phi_theta_mode_0000", and falls back to single‑component names like "phi_0000".

Usage example:
  python plot_flask_dual_eigendec.py \
     --ckpt ../mesh/flask.h5 \
     --evals ./flask_dual_eigenvalues.txt \
     --prefix flask_dual_mode_ \
     --n-show 9 \
     --outdir ./plots_dual

If --evals is omitted, we try "<dirname(ckpt)>/<prefix>eigenvalues.txt" and "eigenvalues.txt".
If --prefix is omitted, we auto-detect from the checkpoint by trying common patterns.
"""

import argparse, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import firedrake as fd

def load_modes(chk_path):
    """Return (mesh, modes, pattern_used).
       Each item of 'modes' is either a MixedFunction (theta,phi) or a scalar Function.
    """
    modes = []
    with fd.CheckpointFile(str(chk_path), "r") as chk:
        # Try to guess mesh name; if multiple, match first in index
        # Common names: "flask", "ismip-c"
        mesh = chk.load_mesh(name="flask")
                break
            except Exception:
                continue
        if mesh is None:
            raise RuntimeError("Could not load mesh from checkpoint (tried 'flask', 'ismip-c').")

        for pat in name_patterns:
            i = 0
            modes.clear()
            while True:
                name = f"{pat}{i:04d}"
                try:
                    f = chk.load_function(mesh, name=name)
                    modes.append(f)
                    i += 1
                except Exception:
                    break
            if modes:
                return mesh, modes, pat
    return mesh, modes, None

def get_components(func, Q=None):
    """Return (theta_comp, phi_comp) as scalar Functions (or (func, None) for single)."""
    # MixedFunction?
    fs = func.function_space()
    try:
        k = fs.num_sub_spaces()
    except Exception:
        k = 1
    if k > 1:
        # Extract subfunctions and project to scalar space Q (if provided)
        th_sub = func.sub(0)
        ph_sub = func.sub(1)
        if Q is None:
            # Use sub spaces if available
            try:
                return fd.Function(th_sub), fd.Function(ph_sub)
            except Exception:
                # project as fallback
                QQ = fs.sub(0)
                P = QQ if Q is None else Q
                return fd.project(th_sub, P), fd.project(ph_sub, P)
        else:
            return fd.project(th_sub, Q), fd.project(ph_sub, Q)
    else:
        return func, None

def symmetric_tripcolor(ax, f, title=None):
    """Plot with diverging colormap centered at 0 for nicer signed structure visualization."""
    vmin = float(f.dat.data_ro.min())
    vmax = float(f.dat.data_ro.max())
    a = max(abs(vmin), abs(vmax))
    norm = mcolors.TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=a)
    from firedrake.pyplot import tripcolor as ftrip
    c = ftrip(f, axes=ax, norm=norm, cmap="RdBu_r")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return c

ckpt ="../mesh/flask.h5"
evals_path ="../results/eigenvalues.txt"
n = 4
outdir="../figures"


ckpt = pathlib.Path(ckpt)
outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)



if evals_path is not None and pathlib.Path(evals_path).exists():
    idx, lam = np.loadtxt(evals_path, unpack=True)
    pos = lam > 0
    neg = lam < 0
    plt.figure(figsize=(7,4.5))
    plt.semilogy(np.nonzero(pos)[0], lam[pos], ".", label="positive")
    if np.any(neg):
        plt.semilogy(np.nonzero(neg)[0], -lam[neg], ".", label="|negative|")
    plt.xlabel("mode index")
    plt.ylabel("eigenvalue magnitude")
    plt.title("K^{-1} H eigenspectrum")
    plt.grid(True, which="both", lw=0.5, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "eigenvalues_semilogy.png", dpi=220)
    plt.close()
    print(f"✓ Wrote {outdir/'eigenvalues_semilogy.png'}")
else:
    print("! eigenvalues file not found (skipping spectrum plot)")

# 2) Load eigenmodes

mesh, raw_modes, pat = load_modes(ckpt)

print(f"Loaded {len(raw_modes)} modes using prefix '{pat}' from {ckpt}")

# Build scalar component lists
# We'll use the Q of a known scalar field for consistent plotting
with fd.CheckpointFile(str(ckpt), "r") as chk:
    # Try to get a scalar space from a known scalar field (e.g., 'friction')
    try:
        C0 = chk.load_function(mesh, name="friction")
        Q = C0.function_space()
    except Exception:
        Q = None

theta_fields, phi_fields = [], []
for f in raw_modes:
    th, ph = get_components(f, Q)
    theta_fields.append(th)
    if ph is not None:
        phi_fields.append(ph)

# 3) Grid plots
from math import ceil
n_show = max(1, min(args.n_show, len(theta_fields)))
cols = 3
rows = int(ceil(n_show/cols))

# Theta panel
fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), constrained_layout=True)
axes = np.asarray(axes).reshape(rows, cols)
for ax in axes.flat[n_show:]:
    ax.axis("off")
for k in range(n_show):
    c = symmetric_tripcolor(axes.flat[k], theta_fields[k], title=f"θ mode {k}")
    fig.colorbar(c, ax=axes.flat[k], shrink=0.8)
fig.suptitle("Leading θ eigenmodes (whitened K^{-1}H)", y=1.02)
fig.savefig(outdir / "leading_theta_modes_grid.png", dpi=220)
plt.close(fig)
print(f"✓ Wrote {outdir/'leading_theta_modes_grid.png'}")

# Phi panel (if present)
if phi_fields:
    n_show_phi = max(1, min(args.n_show, len(phi_fields)))
    rows = int(ceil(n_show_phi/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat[n_show_phi:]:
        ax.axis("off")
    for k in range(n_show_phi):
        c = symmetric_tripcolor(axes.flat[k], phi_fields[k], title=f"φ mode {k}")
        fig.colorbar(c, ax=axes.flat[k], shrink=0.8)
    fig.suptitle("Leading φ eigenmodes (whitened K^{-1}H)", y=1.02)
    fig.savefig(outdir / "leading_phi_modes_grid.png", dpi=220)
    plt.close(fig)
    print(f"✓ Wrote {outdir/'leading_phi_modes_grid.png'}")
else:
    print("! No φ component found; plotted θ only.")

# 4) ParaView series (time slider to browse modes)
# Write first N modes as a timeseries for each component
Npv = min(40, len(theta_fields))
if Npv > 0:
    fth = fd.File(str(outdir / "theta_modes.pvd"))
    for k in range(Npv):
        g = theta_fields[k].copy(deepcopy=True)
        g.rename("theta_mode")
        fth.write(g, time=float(k))
    print(f"✓ Wrote ParaView series: {outdir/'theta_modes.pvd'}")
if phi_fields:
    Npv = min(40, len(phi_fields))
    fph = fd.File(str(outdir / "phi_modes.pvd"))
    for k in range(Npv):
        g = phi_fields[k].copy(deepcopy=True)
        g.rename("phi_mode")
        fph.write(g, time=float(k))
    print(f"✓ Wrote ParaView series: {outdir/'phi_modes.pvd'}")

if __name__ == "__main__":
    main()
