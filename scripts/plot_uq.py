#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

# --------------------------------------------------------------------
# Define input/output files and tags here so they are fixed & reproducible
csv_file  = "../results/uq_results.csv"   # produced by uq_forward_sensitivity_flask.py
outdir    = pathlib.Path("../figures")
tag       = "flask_uq"
ylabel    = "VAF"     # Y-axis label for QoI
T_total   = 20.0      # total years (only needed if converting step_index to time)
nsteps    = 80        # number of steps (only needed if converting step_index to time)
# --------------------------------------------------------------------

outdir.mkdir(parents=True, exist_ok=True)

# ---- load results ----
with open(csv_file, "r") as f:
    header = f.readline().strip().split(",")
header_lower = [h.lower() for h in header]

if header_lower == ["time", "qoi", "sigma"]:
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    x = data[:,0]
    qoi = data[:,1]
    sigma = data[:,2]
    xlabel = "Time (years)"
elif header_lower == ["step_index", "qoi_value", "sigma_q"]:
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    steps = data[:,0]
    qoi   = data[:,1]
    sigma = data[:,2]
    # Convert to time if T_total and nsteps provided
    x = steps * T_total / nsteps
    xlabel = "Time (years)"
else:
    raise ValueError(f"Unrecognized header in {csv_file}: {header}")

# ---- make figure ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8), sharex=True)

# (a) QoI with ±2σ envelope
ax1.plot(x, qoi, color="k", linewidth=1.2, label="QoI")
ax1.fill_between(x, qoi - 2.0*sigma, qoi + 2.0*sigma,
                 alpha=0.4, color="gray", label="±2σ")
ax1.set_ylabel(ylabel)
ax1.set_title("(a) QoI with ±2σ envelope")
ax1.grid(alpha=0.3, linestyle=":", linewidth=0.6)
ax1.legend(frameon=False)

# (b) 2σ time series
ax2.plot(x, 2.0*sigma, linewidth=1.2, color="red")
ax2.set_xlabel(xlabel)
ax2.set_ylabel("2σ")
ax2.set_title("(b) 2σ through time")
ax2.grid(alpha=0.3, linestyle=":", linewidth=0.6)

plt.tight_layout()
figfile = outdir / f"{tag}_qoi_sigma.png"
plt.savefig(figfile, dpi=180)
print(f"[OK] Saved {figfile}")
plt.show()
