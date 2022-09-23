from .constants import G
from .physics import epsilon_pp, epsilon_cno, grad_ad, grad_rad

import jax.numpy as jnp
from functools import partial
from jax import jit

def drdm(r, rho):
    # Derivative of r with respect to m, from mass conservation.
    return 1 / (4 * jnp.pi * r ** 2 * rho)

def dPdm(m, r):
    # Derivative of P with respect to m, from hydrostatic equilibrium.
    return -G * m / (4 * jnp.pi * r ** 4)

def dldm(T, rho, X, Y):
    # Derivative of l with respect to m, from energy conservation.
    return epsilon_cno(rho, T, X, Y) + epsilon_pp(rho, T, X)
    
def dTdm(m, r, l, P, T, kappa):
    # Derivative of T with respect to m, from... thermodynamics.
    g_rad = grad_rad(m, l, P, T, kappa)
    g_ad = grad_ad()
    stable =  g_rad < g_ad
    grad = g_ad
    # if stable:
    #     # @debug("dTdm: stable to convection")
    #     grad = g_rad
    # else:
    #     # @debug("dTdm: unstable to convection")
    #     grad = g_ad
    factor = - G * m * T / (4 * jnp.pi * r ** 4 * P)
    return factor * grad