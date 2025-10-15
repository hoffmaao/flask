import argparse, pathlib, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import geojson
import rasterio
import pygmsh
import firedrake as fd
from firedrake_adjoint import *
import icepack
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
import matplotlib.pyplot as plt
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g, weertman_sliding_law as m
)

def flotationHeight(bed, ρ_I= ρ_I, ρ_W=ρ_W):
    """Given bed elevation, determine height of flotation for function space Q.
    Parameters
    ----------
    zb  : firedrake interp function
        bed elevation (m)
    Q : firedrake function space
        function space
    rho_I : [type], optional
        [description], by default rhoI
    rho_W : [type], optional
        [description], by default rhoW
    Returns
    -------
    zF firedrake interp function
        Flotation height (m)
    """
    # computation for height above flotation
    zF = fd.max_value(-bed * (ρ_W/ρ_I-1), 0)
    return zF

def firedrakeSmooth(q0, α=2e3):
    """[summary]
    Parameters
    ----------
    q0 : firedrake function
        firedrake function to be smooth
    α : float, optional
        parameter that controls the amount of smoothing, which is
        approximately the smoothing lengthscale in m, by default 2e3
    Returns
    -------
    q firedrake interp function
        smoothed result
    """
    q = q0.copy(deepcopy=True)
    J = 0.5 * ((q - q0)**2 + α**2 * fd.inner(fd.grad(q), fd.grad(q))) * fd.dx
    F = fd.derivative(J, q)
    fd.solve(F == 0, q)
    return q


def flotationMask(s, zF, rho_I=ρ_I, rho_W=ρ_W):
    """Using flotation height, create masks for floating and grounded ice.
    Parameters
    ----------
    zF firedrake interp function
        Flotation height (m)
    Q : firedrake function space
        function space
    rho_I : [type], optional
        [description], by default rhoI
    rho_W : [type], optional
        [description], by default rhoW
    Returns
    -------
    floating firedrake interp function
         ice shelf mask 1 floating, 0 grounded
    grounded firedrake interp function
        Grounded mask 1 grounded, 0 floating
    """
    # smooth to avoid isolated points dipping below flotation.

    zAbove = firedrakeSmooth(fd.interpolate(s - zF, s.function_space()), α=100)
    floating = icepack.interpolate(zAbove < 0, Q)
    grounded = icepack.interpolate(zAbove > 0, Q)
    return floating, grounded

def friction_stress(u, C, m):

    r"""Compute the shear stress for a given sliding velocity"""
    return -C * fd.sqrt(fd.inner(u, u)) ** (1 / m - 1) * u


def friction(**kwargs):
    r"""Return the bed friction part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\frac{m}{m + 1}\int_\Omega\tau(u, C)\cdot u\; dx

    where :math:`\\tau(u, C)` is the basal shear stress

    .. math::
       \tau(u, C) = -C|u|^{1/m - 1}u
    """
    u = kwargs["velocity"]
    θ = kwargs["log_friction"]
    m = kwargs["sliding_exponent"]
    C0 = kwargs["friction"]
    C = C0*fd.exp(θ)

    τ = friction_stress(u, C, m)
    return -m / (m + 1) * grounded* fd.inner(τ, u)

def friction_coloumb(**kwargs):
    b = kwargs["bed"]
    s = kwargs["surface"]
    h = kwargs["thickness"]
    u = kwargs["velocity"]
    θ = kwargs["log_friction"]
    C_0 = kwargs["friction"]
    C = C_0 * fd.exp(θ)

    p_W = ρ_W * g * fd.max_value(0, h-s)
    p_I = ρ_I * g * h
    N = grounded * fd.max_value(0, p_I - p_W)
    τ_c = N / 2

    u_c = (τ_c / C) ** m
    u_b = fd.sqrt(fd.inner(u, u))

    return τ_c * (
        (u_c**(1 / m + 1) + u_b**(1 / m + 1))**(m / (m + 1)) - u_c
    )


def viscosity(**kwargs):
    A0 = kwargs['fluidity']
    u = kwargs['velocity']
    h = kwargs['thickness']
    φ = kwargs['log_fluidity']
    A = A0*fd.exp(φ)
    return icepack.models.viscosity.viscosity_depth_averaged(velocity=u,
                                                             thickness=h,
                                                             fluidity=A)

# def friction_weertman(**kwargs):
#     s = kwargs["surface"]
#     h = kwargs["thickness"]
#     u = kwargs["velocity"]
#     θ = kwargs["log_friction"]
#     C = C_0 * firedrake.exp(θ)
    
#     p_W = ρ_W * g * firedrake.max_value(0, h - s)
#     p_I = ρ_I * g * h
#     ϕ = 1 - p_W / p_I

#     return icepack.models.friction.bed_friction(
#         velocity=u,
#         friction=C * ϕ,
#     )


def load_from_checkpoint(chk_path):
    with fd.CheckpointFile(chk_path, "r") as chk:
        mesh = chk.load_mesh(name="flask")
        h    = chk.load_function(mesh, name="thickness")
        s    = chk.load_function(mesh, name="surface")
        b    = chk.load_function(mesh, name="bed")
        u_obs = chk.load_function(mesh, name="velocity_obs")
        σ_obs = chk.load_function(mesh, name="sigma_obs")
        C0 = chk.load_function(mesh, name="friction")
        A0 = chk.load_function(mesh, name="fluidity")

    V = u_obs.function_space()
    Q = C0.function_space()
    return mesh, V, Q, h, s, b, u_obs, σ_obs, C0, A0


def lcurve(mesh, V, Q, h, b, s, u_obs, σ_obs, C0, A0,  # ≈ 1e-16 Pa^-3 yr^-1 in Icepack units
           gammas=(1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,1e1,3e1,1e2,3e2,1e3,3e3,1e4,4e4),
           L = 7.5e3,       # length‑scale (m)
           theta_scale = 1.0, # nondimensional scale for θ
           ):

    # (Re)build solver on the current mesh
    model = icepack.models.IceStream(
        friction=friction, viscosity=viscosity
    )
    opts = {
        "dirichlet_ids": [1, 2, 3, 4, 5],
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            # Use line-search Newton to avoid TR-radius issues in adjoint
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9,
            "snes_atol": 1e-9,
            "snes_stol": 0.0,
            "snes_max_it": 80,
            "snes_error_if_not_converged": True,

            # If you want output, use flags (None), not True:
            # "snes_monitor": None,
            # "snes_converged_reason": None,

            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",

            # Optional KSP monitors (also flags):
            # "ksp_converged_reason": None,
            # "ksp_monitor": None,
        }
}

    solver=icepack.solvers.FlowSolver(model, **opts)

    # domain area for normalisation
    area = fd.Constant(fd.assemble(fd.Constant(1.0) * fd.dx(mesh)),domain=mesh)
    θ = fd.Function(Q)
    φ = fd.Function(Q)

    zF=flotationHeight(b)
    floating, grounded=flotationMask(s,zF)



    def simulation(controls):
        return solver.diagnostic_solve(
            velocity=u_obs,
            thickness=h,
            surface=s,
            log_friction=controls[0], 
            log_fluidity=controls[1],
            friction = C0,
            fluidity = A0,
            bed = b,
            sliding_exponent=fd.Constant(1.0,domain=mesh)
            )

    def loss_function(u):
        du = u - u_obs
        return 0.5 / area * ((du[0]/σ_obs[0])**2 + (du[1]/σ_obs[1])**2) * fd.dx

    #def regularization(controls):
    #    return 0.5 / area  * gamma_1 * (L / theta_scale)**2 * fd.inner(fd.grad(controls[0]), fd.grad(constrols[0])) * fd.dx +
    #    0.5 / area  * gamma_2 * (L / theta_scale)**2 * fd.inner(fd.grad(controls[1]), fd.grad(constrols[1])) * fd.dx



    points = []
    gamma_2 = fd.Constant(1,domain=mesh)
    for g in gammas:
        gamma_1 = fd.Constant(g,domain=mesh)
        θ = fd.Function(Q)
        φ = fd.Function(Q)

        def regularization(controls):
            return 0.5 / area  * gamma_1 * (L / theta_scale)**2 * fd.inner(fd.grad(controls[0]), fd.grad(controls[0])) * fd.dx + \
            0.5 / area  * gamma_2 * (L / theta_scale)**2 * fd.inner(fd.grad(controls[1]), fd.grad(controls[1])) * fd.dx
        
        problem = StatisticsProblem(
            simulation=simulation,
            loss_functional=loss_function,
            regularization=regularization,
            controls=[θ,φ]
        )
        
        estimator = MaximumProbabilityEstimator(
            problem,
            gradient_tolerance=1e-8,
            step_tolerance=1e-8,
            max_iterations=400,
        )
        θ,φ = estimator.solve()
        u = simulation([θ,φ])
        # evaluate terms for L‑curve
        mis = fd.assemble(loss_function(u))
        reg = fd.assemble(regularization([θ,φ]))
        print(reg)
        print(mis)
        # save result for this γ
        with fd.CheckpointFile("../mesh/flask.h5", "a") as chk:
            chk.save_mesh(mesh)             # for viewing; periodic IDs aren’t preserved
            chk.save_function(θ, name=f"gamma{g:.2e}_log_friction")
            chk.save_function(φ, name=f"gamma{g:.2e}_log_fluidity")
            C = fd.Function(Q, name=f"gamma{g:.2e}_beta"); C.interpolate(C0*fd.exp(θ))
            chk.save_function(C, name=f"gamma{g:.2e}_friction")
            A = fd.Function(Q, name=f"gamma{g:.2e}_beta"); A.interpolate(A0*fd.exp(φ))
            chk.save_function(A, name=f"gamma{g:.2e}_fluidity")
            chk.save_function(u, name=f"gamma{g:.2e}_velocity")

        points.append((g, mis, reg))
        print(f"[γ={g:.2e}] misfit={mis:.6e}, regularizer={reg:.6e}")

    # L‑curve plot (misfit vs regularizer) on log–log axesfiredrake.
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    plt.figure(figsize=(6,5))
    plt.loglog(xs, ys, marker="o")
    for (g,x,y) in zip(gammas, xs, ys):
        plt.annotate(f"{g:.0e}", (x,y), textcoords="offset points", xytext=(4,4), fontsize=8)
    plt.xlabel("velocity misfit term")
    plt.ylabel("regularization term")
    plt.title("L‑curve (β inversion, Icepack statistics)")
    plt.tight_layout()
    plt.savefig("../figures/lcurve.png", dpi=200)
    with open("../results/lcurve.csv", "w") as f:
        f.write("gamma,misfit,regularization\n")
        for g,m,r in points:
            f.write(f"{g},{m},{r}\n")
    return points

mesh, V, Q, h, s, b, u_obs, σ_obs, C0, A0 = load_from_checkpoint("../mesh/flask.h5")

zF=flotationHeight(b)
floating, grounded=flotationMask(s,zF)

model = icepack.models.IceStream(
        friction=friction, viscosity=viscosity
    )  # defaults: depth-averaged viscosity, Weertman friction

# 2) pick *explicit* BC markers from the mesh (update these!)
opts = {
        "dirichlet_ids": [1, 2, 3, 4, 5],
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            # Use line-search Newton to avoid TR-radius issues in adjoint
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9,
            "snes_atol": 1e-12,
            "snes_stol": 0.0,
            "snes_max_it": 80,
            "snes_error_if_not_converged": True,

            # If you want output, use flags (None), not True:
            # "snes_monitor": None,
            # "snes_converged_reason": None,

            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",

            # Optional KSP monitors (also flags):
            # "ksp_converged_reason": None,
            # "ksp_monitor": None,
        }
}

solver=icepack.solvers.FlowSolver(model, **opts)

θ0 = fd.Function(Q)  # zeros
φ0 = fd.Function(Q)
u0 = solver.diagnostic_solve(
            velocity=u_obs,
            thickness=h,
            surface=s,
            log_friction=θ0, 
            log_fluidity=φ0,
            friction = C0,
            fluidity = A0,
            bed = b,
            sliding_exponent=fd.Constant(1.0,domain=mesh)
            )
speed = fd.project(fd.sqrt(fd.inner(u0, u0)), Q)
arr = speed.dat.data_ro
print("Forward OK? NaNs:", np.isnan(arr).any(), "min|u|:", float(arr.min()), "max|u|:", float(arr.max()))

outline_filename = "../mesh/flask.geojson"
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)


coords = np.array(list(geojson.utils.coords(outline)))
delta = 30e3
xmin, xmax = coords[:, 0].min() - delta, coords[:, 0].max() + delta
ymin, ymax = coords[:, 1].min() - delta, coords[:, 1].max() + delta


image_filename = icepack.datasets.fetch_mosaic_of_antarctica()
with rasterio.open(image_filename, "r") as image_file:
    height, width = image_file.height, image_file.width
    transform = image_file.transform
    window = rasterio.windows.from_bounds(
        left=xmin,
        bottom=ymin,
        right=xmax,
        top=ymax,
        transform=transform,
    )
    image = image_file.read(indexes=1, window=window, masked=True)



def subplots(*args, **kwargs):
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    xmin, ymin, xmax, ymax = rasterio.windows.bounds(window, transform)
    axes.imshow(
        image,
        cmap="Greys_r",
        vmin=12e3,
        vmax=19.38e3,
        extent=(xmin, xmax, ymin, ymax),
    )
    axes.tick_params(labelrotation=25)

    return fig, axes

fig, axes = subplots()
colors = fd.tripcolor(speed, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig('../figures/speed_test.png')

gamma = "1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,1e1,3e1,1e2,3e2"

gammas = tuple(float(v) for v in gamma.split(","))

lcurve(mesh, V, Q, h, b, s, u_obs, σ_obs, C0, A0, 
    gammas=gammas, L=fd.Constant(7.5e3,domain=mesh), 
    theta_scale=fd.Constant(1.0,domain=mesh))

