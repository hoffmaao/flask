#!/usr/bin/env python3
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import firedrake as fd

# ---- paths (fixed, per request) ----
EVALS = pathlib.Path("../results/eigenvalues.txt")
CKPT  = pathlib.Path("../mesh/flask.h5")
PLOTS = pathlib.Path("../figures")

# -----------------------------
# 1) Eigenvalue spectrum (semilogy)
# -----------------------------
if not EVALS.exists():
    raise FileNotFoundError(f"Eigenvalues file not found: {EVALS}")
try:
    idx, lam = np.loadtxt(EVALS, unpack=True)
except ValueError:
    # fallback if only values are present
    lam = np.loadtxt(EVALS)
    idx = np.arange(len(lam))

pos = lam > 0
neg = lam < 0

plt.figure(figsize=(7,4.5))
plt.semilogy(np.where(pos)[0], lam[pos], ".", label="positive")
if np.any(neg):
    plt.semilogy(np.where(neg)[0], -lam[neg], ".", label="|negative|")
plt.xlabel("mode index")
plt.ylabel("eigenvalue magnitude")
plt.title("Whitened K$^{-1}$H eigenspectrum")
plt.grid(True, which="both", lw=0.5, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "eigenvalues_semilogy.png", dpi=220)
plt.close()
print(f"✓ wrote {PLOTS/'eigenvalues_semilogy.png'}")

# -----------------------------
# 2) Load eigenmodes from checkpoint
# -----------------------------
if not CKPT.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {CKPT}")
with fd.CheckpointFile(str(CKPT), "r") as chk:
    # try mesh name 'flask', else default
    try:
        mesh = chk.load_mesh(name="flask")
    except Exception:
        mesh = chk.load_mesh()
    modes = []
    i = 0
    while True:
        name = f"mode_{i:04d}"
        try:
            f = chk.load_function(mesh, name=name)
            modes.append(f)
            i += 1
        except Exception:
            break

if not modes:
    raise RuntimeError("No modes found in checkpoint using prefix 'mode_{i:04d}'")

print(f"Loaded {len(modes)} modes from {CKPT}")

# -----------------------------
# 3) Plot leading modes (scalar or mixed)
# -----------------------------
from firedrake.pyplot import tripcolor as ftrip

def symmetric_tripcolor(ax, f, title=None):
    vmin = float(f.dat.data_ro.min())
    vmax = float(f.dat.data_ro.max())
    a = max(abs(vmin), abs(vmax))
    if a == 0.0:
        a = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=a)
    c = ftrip(f, axes=ax, norm=norm, cmap="RdBu_r")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return c

def num_sub_spaces(func):
    try:
        return func.function_space().num_sub_spaces()
    except Exception:
        return 1

kshow = min(9, len(modes))

if num_sub_spaces(modes[0]) == 1:
    # Scalar modes
    cols = 3
    rows = int(np.ceil(kshow/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat[kshow:]:
        ax.axis("off")
    for k in range(kshow):
        c = symmetric_tripcolor(axes.flat[k], modes[k], title=f"mode {k}")
        fig.colorbar(c, ax=axes.flat[k], shrink=0.8)
    fig.suptitle("Leading eigenmodes (scalar)", y=1.02)
    fig.savefig(PLOTS / "leading_modes_grid.png", dpi=220)
    plt.close(fig)
    print(f"✓ wrote {PLOTS/'leading_modes_grid.png'}")


else:
    # MixedFunction modes: split into components (theta, phi)
    theta_list, phi_list = [], []
    # choose a scalar Q space from the first component for projection if needed
    Q0 = modes[0].sub(0).function_space()
    for f in modes[:]:
        th = fd.Function(Q0); th.assign(f.sub(0))
        ph = fd.Function(Q0); ph.assign(f.sub(1))
        theta_list.append(th)
        phi_list.append(ph)

    # θ grid
    cols = 3
    rows = int(np.ceil(kshow/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat[kshow:]:
        ax.axis("off")
    for k in range(kshow):
        c = symmetric_tripcolor(axes.flat[k], theta_list[k], title=f"θ mode {k}")
        fig.colorbar(c, ax=axes.flat[k], shrink=0.8)
    fig.suptitle("Leading θ eigenmodes", y=1.02)
    fig.savefig(PLOTS / "leading_theta_modes_grid.png", dpi=220)
    plt.close(fig)
    print(f"✓ wrote {PLOTS/'leading_theta_modes_grid.png'}")

    # φ grid
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat[kshow:]:
        ax.axis("off")
    for k in range(kshow):
        c = symmetric_tripcolor(axes.flat[k], phi_list[k], title=f"φ mode {k}")
        fig.colorbar(c, ax=axes.flat[k], shrink=0.8)
    fig.suptitle("Leading φ eigenmodes", y=1.02)
    fig.savefig(PLOTS / "leading_phi_modes_grid.png", dpi=220)
    plt.close(fig)
    print(f"✓ wrote {PLOTS/'leading_phi_modes_grid.png'}")


