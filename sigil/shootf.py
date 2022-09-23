from .constants import Msun, Rsun, Lsun, m_H, k_B
from .physics import mu_from_composition, rho_ideal, T_surface, P_surface
from .equations import *
from .opacity import opacity

from numpy import genfromtxt
op = genfromtxt('data/opacity/solar.txt', invalid_raise=False)

logT = op[:,0]
logR = jnp.arange(-8, 1.5, step=0.5)
logkappa = op[:, 1:]

def init_guess(M, X, Y):
    # #=
    # Gives a reasonable guess for Pc, Tc, R, and L for a star of mass M
    # =#
    yhalf = [Rsun * 10**0.062544715341505616, Lsun * 10**-0.36630780815983022, 10**19.700356657315954, 10**7.2863105891013289]
    y2 = [Rsun * 10**0.36072469919806871, Lsun * 10**1.3021335874028637, 10**17.186879800538229, 10**7.3557279873577759]
    y5 = [Rsun * 10**0.59322648100945885, Lsun * 10**2.8743140385067893, 10**16.752833342638592, 10**7.4547058342916337]
    y4 = [Rsun * 10**0.52962615387538248, Lsun * 10**2.50275963393719, 10**16.855407680647915, 10**7.429616124439673]
    y10 = [Rsun * 10**0.68191099104456998, Lsun * 10**3.8447263619458569, 10**16.517451455410988, 10**7.5012272142698366] 
    if M == Msun:
        return [Rsun, Lsun, 2.4e17, 1.6e7]
    elif M == 0.5 * Msun:
        return yhalf
    elif M == 2 * Msun:
        return y2
    elif (M >= 4 * Msun) & (M <= 5 * Msun):
        dm = M - 4 * Msun
        dydm = (y5 - y4) / Msun
        return y4 + dydm * dm
    elif M == 10 * Msun:
        return y10
    mu = mu_from_composition(X, Y)
    
    # mass-radius relation
    if M / Msun > 1.3: # then CNO burning dominates
        z1 = 0.8
        z2 = 0.64
    else: # then pp chain dominates
        z1 = 0.5
        z2 = 0.1
    R = Rsun * (M / Msun)**z1 * (mu / 0.61)**z2
    # mass-luminosity relation
    L = Lsun * (M / Msun)**3 * (mu / 0.61)**4
    # Pc, Tc guesses, essentially dimensional analysis with fudges
    Pc = 2 * G * M**2 / (jnp.pi * R**4)
    Tc = 8 * 0.02 * mu * m_H * G * M / (k_B * R)
    return [R, L, Pc, Tc]

def load1(m, Pc, Tc, X, Y):
    # #=
    # Subroutine for shootf.
    # Loads the (r, l, P, T) center (m = 0) values for starting the outward 
    # integration.  
    # =#
    mu = mu_from_composition(X, Y)
    # m = 0.01 * Msun # needs tuning
    rhoc = rho_ideal(Pc, Tc, mu)
    rc = (3 * m / (4 * jnp.pi * rhoc))**(1 / 3)
    lc = dldm(Tc, rhoc, X, Y) * m
    return [rc, lc, Pc, Tc]

def load2(M, Rs, Ls, X, Y):
    # #=
    # Subroutine for shootf.
    # Loads the (r, l, P, T) surface (m = M) values for starting the inward
    # integration.
    # =#
    mu = mu_from_composition(X, Y)
    converged = False
    rho_guess = 1e-8 # need a less arbitrary guess?
    Ts = T_surface(Rs, Ls)
    Ps = 0
    count = 1
    while ~converged & (count < 100):
        kappa = opacity(rho_guess, Ts, logT, logR, logkappa)
        Ps = P_surface(M, Rs, kappa)
        rho = rho_ideal(Ps, Ts, mu)
        relerr = (rho - rho_guess) / rho
        if abs(relerr) < 1e-6: # need less arbitrary convergence condition
            converged = True
        else:
            rho_guess = rho_guess + relerr * rho / 6
            count += 1
    return [Rs, Ls, Ps, Ts]

# def profiles(m, M, Rs, Ls, Pc, Tc, X, Y):
#     # #=
#     # Integrates outward from m to mf and inward from M to mf to construct
#     # r, l, P, and T profiles from the given initial conditions.
#     # =#
#     yc = load1(m, Pc, Tc, X, Y)
#     prob = ODEProblem(deriv!,yc,(0, mf), (X, Y))
#     sol_out = solve(prob, Tsit5());
#     sol_out_v = hcat(sol_out.u...)'

#     ys = load2(M, Rs, Ls, X, Y)
#     prob = ODEProblem(deriv!,ys,(M, mf), [X, Y])
#     sol_in = solve(prob, Tsit5());
#     sol_in_v = hcat(sol_in.u...)'
#     return sol_out.t, sol_in.t, sol_out_v, sol_in_v

# # def profiles(star::Star, Rs, Ls, Pc, Tc; n=1000)
# #     return profiles(star.m, star.M, Rs, Ls, Pc, Tc, star.X, star.Y, star.mf)
# # end

# def deriv(du,u,p,m):
#     r, l, P, T = u
#     X, Y = p
    
#     mu = mu_from_composition(X, Y)
#     rho = rho_ideal(P, T, mu)
    
#     kappa = opacity(logkappa_spl, rho, T)
    
#     du[1] = drdm(m, r, l, P, T, rho, mu)
#     du[2] = dldm(m, r, l, P, T, rho, X, Y)
#     du[3] = dPdm(m, r, l, P, T)
#     du[4] = dTdm(m, r, l, P, T, mu, kappa)
#     return du

# def score(m, M, Rs, Ls, Pc, Tc, X, Y, mf):
#     # #=
#     # def to zero. 
#     # =#
#     try:
#         sol_out_t, sol_in_t, sol_out_v, sol_in_v = profiles(m, M, Rs, Ls, Pc, Tc, X, Y, mf)
#         ss = (sol_out_v[end, :] - sol_in_v[end, :]) / sol_in_v[end, :]
#     catch
#         [NaN]
#     end
    
#     return (sum(ss.^2))

# def shootf(m, M, X, Y; fixed_point=0.8)
#     #= 
#     Shooting to a fixed point
#     =#
#     y0 = init_guess(M, X, Y)
#     mf = fixed_point * M
#     f(y) = score(m, M, y[1], y[2], y[3], y[4], X, Y, mf)
    
#     res = optimize(f, y0,
#                 g_tol = 1e-15,
#                 iterations = 10000,
#                 allow_f_increases=true,
#                 store_trace = false,
#                 show_trace = false
#     )
#     return res
# end # shootf

# def shootf(star::Star)
#     return shootf(star.m, star.M, star.X, star.Y)
# end

# def shootf!(star::Star)
#     res = shootf(star)
#     bc = Optim.minimizer(res)
#     sol_out_t, sol_in_t, sol_out_v, sol_in_v = profiles(star, bc...)
#     star.grid = vcat(sol_out_t, reverse(sol_in_t))
#     star.r = vcat(sol_out_v[:,1], reverse(sol_in_v[:,1]))
#     star.l = vcat(sol_out_v[:,2], reverse(sol_in_v[:,2]))
#     star.P = vcat(sol_out_v[:,3], reverse(sol_in_v[:,3]))
#     star.T = vcat(sol_out_v[:,4], reverse(sol_in_v[:,4]));
#     return res
# end