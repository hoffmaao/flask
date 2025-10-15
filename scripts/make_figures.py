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
    θ = kwargs["log_friction"]
    m = kwargs["sliding_exponent"]
    C0 = kwargs["friction"]
    C = C0*firedrake.exp(θ)

    τ = friction_stress(u, C, m)
    return -m / (m + 1) * grounded* firedrake.inner(τ, u)

def load_from_checkpoint(chk_path,gamma=3.00e-01):
    with firedrake.CheckpointFile(chk_path, "r") as chk:
        mesh = chk.load_mesh(name="flask")
        h    = chk.load_function(mesh, name="thickness")
        s    = chk.load_function(mesh, name="surface")
        b    = chk.load_function(mesh, name="bed")
        u_obs = chk.load_function(mesh, name="velocity_obs")
        σ_obs = chk.load_function(mesh, name="sigma_obs")
        C0 = chk.load_function(mesh, name="friction")
        A0 = chk.load_function(mesh, name="fluidity")
        θ = chk.load_function(mesh, name=f"gamma{g:.2e}_log_friction")
        φ = chk.load_function(mesh, name=f"gamma{g:.2e}_log_fluidity")
        u = chk.load_function(mesh, name=f"gamma{g:.2e}_velocity")

    V = u_obs.function_space()
    Q = C0.function_space()
    C = firedrake.Function(Q, name=f"gamma{g:.2e}_friction"); C.interpolate(C0*firedrake.exp(θ))
    A = firedrake.Function(Q, name=f"gamma{g:.2e}_fluidity"); A.interpolate(A0*firedrake.exp(φ))
    return mesh, V, Q, h, s, b, u_obs, σ_obs, C0, A0, θ, φ, A, C, u

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

g = 3.00e-03

mesh, V, Q, h, s, b, u_obs, σ_obs, C0, A0, θ, φ, A, C, u = load_from_checkpoint("../mesh/flask.h5", gamma = g)



zF=flotationHeight(b)
floating, grounded=flotationMask(s,zF)

U = sqrt(inner(u, u))
τ_b = firedrake.interpolate(C * grounded * U, Q)




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



fig, axes = subplots()
colors = firedrake.tripcolor(C, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig(f'../figures/friction_gamma{g:.2e}.png', dpi=300)

fig, axes = subplots()
colors = firedrake.tripcolor(A, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig(f'../figures/fluidity_gamma{g:.2e}.png', dpi=300)

fig, axes = subplots()
colors = firedrake.tripcolor(τ_b, vmin=0, vmax=.3, axes=axes,alpha=.5)
fig.colorbar(colors);
fig.savefig(f'../figures/shear_stress_gamma{g:.2e}.png', dpi=300)