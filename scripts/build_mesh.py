import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import geojson
import rasterio
import pygmsh
import firedrake
from firedrake import Constant, sqrt, inner, grad, dx
import icepack
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g, weertman_sliding_law as m
)


outline_filename = "../mesh/flask.geojson"
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

gmsh_mesh = icepack.meshing.collection_to_gmsh(outline)
gmsh_mesh.write("../mesh/flask.msh")
base = firedrake.Mesh("../mesh/flask.msh",name="flask")
hier = firedrake.MeshHierarchy(base,2)
mesh = hier[-1]
mesh.name="flask"
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
kwargs = {
    "interior_kw": {"linewidth": 0.25},
    "boundary_kw": {"linewidth": 2},
}
firedrake.triplot(mesh, axes=axes, **kwargs)
axes.legend()
fig.savefig('../figures/mesh.png')


Q = firedrake.FunctionSpace(mesh, family="CG", degree=2)
V = firedrake.VectorFunctionSpace(mesh, family="CG", degree=2)


bedmachine_filename = icepack.datasets.fetch_bedmachine_antarctica()
thickness_filename = f"netcdf:{bedmachine_filename}:thickness"
with rasterio.open(thickness_filename, "r") as thickness_file:
    h = icepack.interpolate(thickness_file, Q)
    
bed_filename = f"netcdf:{bedmachine_filename}:bed"
with rasterio.open(bed_filename, "r") as bed_file:
    bed = icepack.interpolate(bed_file, Q)

s = icepack.compute_surface(thickness=h, bed=bed)

fig, axes = subplots()
colors = firedrake.tripcolor(s, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig('../figures/surface.png')

fig, axes = subplots()
colors = firedrake.tripcolor(h, axes=axes,alpha=.5)
fig.colorbar(colors,label="thickness (m)");
fig.savefig('../figures/thickness.png')


measures_filename = icepack.datasets.fetch_measures_antarctica()
with rasterio.open(f"netcdf:{measures_filename}:VX", "r") as vx_file, \
     rasterio.open(f"netcdf:{measures_filename}:VY", "r") as vy_file:
    u_obs = icepack.interpolate((vx_file, vy_file), V)

with rasterio.open(f"netcdf:{measures_filename}:ERRX", "r") as ex_file, \
     rasterio.open(f"netcdf:{measures_filename}:ERRY", "r") as ey_file:
    σ_obs = icepack.interpolate((ex_file, ey_file), V)

log_norm = matplotlib.colors.LogNorm(vmin=1, vmax=200.0)

fig, axes = subplots()
streamlines = firedrake.streamplot(
    u_obs, norm=log_norm, axes=axes, resolution=2.5e3, seed=1729
)
fig.colorbar(streamlines,label="velocity (m/yr)");
fig.savefig('../figures/velocity.png')


C0 = firedrake.Function(Q); C0.interpolate(Constant(5e-3))

T = firedrake.Constant(262.0)
A0 = firedrake.Function(Q); A0.interpolate(icepack.rate_factor(T))

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
    zF = firedrake.max_value(-bed * (ρ_W/ρ_I-1), 0)
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
    J = 0.5 * ((q - q0)**2 + α**2 * firedrake.inner(firedrake.grad(q), firedrake.grad(q))) * firedrake.dx
    F = firedrake.derivative(J, q)
    firedrake.solve(F == 0, q)
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

    zAbove = firedrakeSmooth(firedrake.interpolate(s - zF, s.function_space()), α=100)
    floating = icepack.interpolate(zAbove < 0, Q)
    grounded = icepack.interpolate(zAbove > 0, Q)
    return floating, grounded


def friction_stress(u, C, m):

    r"""Compute the shear stress for a given sliding velocity"""
    return -C * firedrake.sqrt(firedrake.inner(u, u)) ** (1 / m - 1) * u


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
    m = kwargs["sliding_exponent"]
    C = kwargs["friction"]
    p_W = ρ_W * g * firedrake.max_value(0, -(s - h))
    p_I = ρ_I * g * h
    

    τ = friction_stress(u, C, m)
    return -m / (m + 1) * grounded * firedrake.inner(τ, u)

def friction_coloumb(**kwargs):
    s = kwargs["surface"]
    h = kwargs["thickness"]
    u = kwargs["velocity"]
    C0 = kwargs["friction"]
    C = C0
    
    p_W = ρ_W * g * firedrake.max_value(0, -(s - h))
    p_I = ρ_I * g * h
    N = firedrake.max_value(0, p_I - p_W)
    τ_c = N / 2

    u_c = (τ_c / C) ** m
    u_b = firedrake.sqrt(firedrake.inner(u, u))

    return τ_c * (
        (u_c**(1 / m + 1) + u_b**(1 / m + 1))**(m / (m + 1)) - u_c
    )

def InitWeertman(s, h, u, grounded, u_0=.1, τ_0=100.0, rho_I=ρ_I):
    """Compute intitial beta using 0.5 taud.
    Parameters
    ----------
    s : firedrake function
        model surface elevation
    h : firedrake function
        model thickness
    V : firedrake vector function space
        vector function space
    Q : firedrake function space
        scalar function space
    grounded : firedrake function
        Mask with 1s for grounded 0 for floating.
    """
    u=firedrakeSmooth(u,α=3000)
    s=firedrakeSmooth(s,α=3000)
    h=firedrakeSmooth(h,α=3000)
    Q = h.function_space()
    V = u.function_space()
    f = -rho_I * g * h * inner(grad(s), u) / firedrake.sqrt(inner(u, u))
    τ_d = firedrake.Function(Q)
    L = 5e3
    J = 0.5 * ((τ_d - f)**2 + L**2 * inner(grad(τ_d), grad(τ_d))) * dx

    firedrake.solve(firedrake.derivative(J, τ_d) == 0, τ_d,
                    solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    fraction = firedrake.Constant(0.75)
    stress = firedrake.max_value(firedrake.sqrt(firedrake.inner(τ_d, τ_d)),firedrake.Constant(0.0001))
    U = firedrake.max_value(firedrake.sqrt(inner(u, u)), 1)
    C0 = firedrake.interpolate(fraction * stress / U,Q)

    print('stress', firedrake.assemble(stress * firedrake.dx))
    # = firedrake.interpolate(-firedrake.ln(C), Q)
    return C0


zF=flotationHeight(bed)
floating, grounded=flotationMask(s,zF)
C0 = InitWeertman(s,h,u_obs,grounded)

fig, axes = subplots()
colors = firedrake.tripcolor(C0, axes=axes,alpha=.5)
fig.colorbar(colors,label=r"resistance (Pa $\cdot$ m/yr$^{-1}$)");
fig.savefig('../figures/friction.png')


model = icepack.models.IceStream(friction=friction)
opts = {
    "dirichlet_ids": [1,2,3,4,5],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    "prognostic_solver_parameters": {
        "ksp_type": "gmres",
        "pc_type": "ilu",
    },
}
solver = icepack.solvers.FlowSolver(model, **opts)


u = solver.diagnostic_solve(
    velocity=u_obs,
    thickness=h,
    surface=s,
    fluidity=A0,
    friction=C0,
    sliding_exponent=Constant(1.,domain=mesh)
)

fig, axes = subplots()
streamlines = firedrake.streamplot(
    u, norm=log_norm, axes=axes, resolution=2.5e3, seed=1729
)
fig.colorbar(streamlines,label="velocity (m/yr)");
fig.savefig('../figures/velocity_solution.png')


with firedrake.CheckpointFile("../mesh/flask.h5", "w") as checkpoint:
    checkpoint.save_mesh(mesh)
    checkpoint.save_function(C0, name="friction")
    checkpoint.save_function(A0, name="fluidity")
    checkpoint.save_function(u, name="velocity")
    checkpoint.save_function(u_obs, name="velocity_obs")
    checkpoint.save_function(σ_obs, name="sigma_obs")
    checkpoint.save_function(bed, name="bed")
    checkpoint.save_function(s, name="surface")
    checkpoint.save_function(h, name="thickness")