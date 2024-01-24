import numpy as np
from scipy.interpolate import RegularGridInterpolator

from utils import lumfun_nonpar_grid

ZMIN = 3.0


def xlf(model="pde", fabs=True):
    if fabs:
        model = globals()[f"_{model}_fabs"]
    else:
        model = globals()[f"_{model}"]

    return model


# def _pde_fabs(z, loglx, lognh, lognorm, loglstar, gamma1, gamma2, pden, c, psi3, fctk, zmin=ZMIN):
#     return _pde(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, zmin) * fabs(z, loglx, lognh, c, psi3, fctk)

def _pde_fabs(z, loglx, lognh, lognorm, loglstar, gamma1, gamma2, pden, logepsilon, logfctk, psi3, c, a2, zmin=ZMIN):
    f = fabs(z, loglx, lognh, logepsilon, logfctk, psi3, c, a2)
    return _pde(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, zmin) * f

def _pde(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, zmin=ZMIN):
    # Pure density evolution luminosity function
    e_den = _evolution_function(z, pden, zmin)
    return _double_powerlaw(10**loglx, 10**lognorm * e_den, 10**loglstar, gamma1, gamma2)


def _ldde_fabs(z, loglx, lognh, lognorm, loglstar, gamma1, gamma2, pden, beta, logepsilon, logfctk, psi3, c, a2, zmin=ZMIN):
    f = fabs(z, loglx, lognh, logepsilon, logfctk, psi3, c, a2)
    return _ldde(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, beta, zmin) * f

def _ldde(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, beta, zmin=ZMIN):
    # Luminosity dependent density evolution luminosity function
    # (Hasinger+2005)
    p = pden + beta * (loglx - 44)
    e_den = _evolution_function(z, p, zmin)
    return _double_powerlaw(10**loglx, 10**lognorm * e_den, 10**loglstar, gamma1, gamma2)


def _lade_fabs(z, loglx, lognh, lognorm, loglstar, gamma1, gamma2, pden, plum, logepsilon, logfctk, psi3, c, a2, zmin=ZMIN):
    f = fabs(z, loglx, lognh, logepsilon, logfctk, psi3, c, a2)
    return _lade(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, plum, zmin) * f

def _lade(z, loglx, lognorm, loglstar, gamma1, gamma2, pden, plum, zmin=ZMIN):
    # Luminosity and density evolution luminosity function
    # (Vito+2014, Georgakakis+2015)
    e_den = 10**(pden * (z - zmin))
    e_lum = _evolution_function(z, plum, zmin)
    return _double_powerlaw(10**loglx, 10**lognorm * e_den, 10**loglstar * e_lum, gamma1, gamma2)


def _ueda(z, loglx, lognorm, loglstar, gamma1, gamma2, logla1, a1, zc1star, p1star, beta1):
    # Ueda+2003, Ueda+2014
    p1 = p1star + beta1 * (loglx - 44)
    p2 = -1.5
    p3 = -6.2
    
    zc2star = 3.0
    logla2 = 44
    a2 = -0.1
    
    mask_lx = loglx <= logla1
    zc1 = zc1star * np.ones_like(loglx)    
    zc1[mask_lx] = zc1star * (10**(loglx[mask_lx] - logla1))**a1

    mask_lx = loglx <= logla2
    zc2 = zc2star * np.ones_like(loglx)    
    zc2[mask_lx] = zc2star * (10**(loglx[mask_lx] - logla2))**a2
    
    
    ezlx1 = np.logical_and(zc1 < z, z <= zc2)
    ezlx2 = z > zc2

    ezlx = (1 + z)**p1
    ezlx[ezlx1] = (1 + zc1[ezlx1])**p1[ezlx1] * _evolution_function(z[ezlx1], p2, zc1[ezlx1])
    ezlx[ezlx2] = (1 + zc1[ezlx2])**p1[ezlx2] * _evolution_function(zc2[ezlx2], p2, zc1[ezlx2]) * _evolution_function(z[ezlx2], p3, zc2[ezlx2])  
    
    return _double_powerlaw(10**loglx, 10**lognorm * ezlx, 10**loglstar, gamma1, gamma2)


def _evolution_function(z, p, zmin):
    return ((1 + z) / (1 + zmin))**p


def _double_powerlaw(lx, norm, lstar, gamma1, gamma2):
    return norm / ((lx / lstar)**gamma1 + (lx / lstar)**gamma2)


# def fabs(z, loglx, lognh, beta, psi3, fctk):     
#     nh2023 = np.logical_and(lognh >= 20, lognh < 23)
#     nh2324 = np.logical_and(lognh >= 23, lognh < 24)
#     nh2426 = np.logical_and(lognh >= 24, lognh <=26)
    
#     f = np.zeros_like(z)
#     f[nh2023] = 1 - _psi(loglx[nh2023], z[nh2023], beta, psi3)
#     f[nh2324] = 1 / (1 + fctk) * _psi(loglx[nh2324], z[nh2324], beta, psi3)
#     f[nh2426] = fctk / (1 + fctk) * _psi(loglx[nh2426], z[nh2426], beta, psi3)

#     return f


# def _psi(loglx, z, beta, psi3, psi_min=0.2, psi_max=0.99):
#     maxx = np.maximum(_psi43p75(z, psi3) - beta*(loglx - 43.75), psi_min)
#     psi = np.minimum(psi_max, maxx)

#     return psi


# def _psi43p75(z, psi3, psi_local=0.43, psi_mid=0.5, a1=0.48):
#     zlow = z < 2
#     zup = z >=3

#     psi4375 = psi_mid * np.ones_like(z)
#     psi4375[zlow] = psi_local * (1 + z[zlow])**a1 / 2
#     psi4375[zup] = psi3 

#     return psi4375



def fabs(z, loglx, lognh, logepsilon=0.34, logfctk=1.0, psi3=1.0, beta=0.24, a2=0):
    nh2023 = np.logical_and(lognh >= 20, lognh < 23)
    nh2324 = np.logical_and(lognh >= 23, lognh < 24)
    nh2426 = np.logical_and(lognh >= 24, lognh <=26)
    
    fctk = 10**logfctk
    epsilon = 10**logepsilon
    ll = epsilon / (1 + epsilon)

    f = np.zeros_like(z)

    if ll + fctk >= 1:
        f[nh2023] = 1e-99
    else:
        f[nh2023] = (1 - (ll + fctk) * _psi(loglx[nh2023], z[nh2023], psi3, beta, a2)) / 3
    
    f[nh2324] = ll * _psi(loglx[nh2324], z[nh2324], psi3, beta, a2)
    f[nh2426] = fctk * _psi(loglx[nh2426], z[nh2426], psi3, beta, a2) / 2

    return f


def _psi(loglx, z, psi3, beta, a2, psi_min=0.2, psi_max=0.99):
    maxx = np.maximum(_psi43p75(z, psi3, a2) - beta*(loglx - 43.75), psi_min)
    psi = np.minimum(psi_max, maxx)

    return 1.0 #psi


def _psi43p75(z, psi3, a2, psi2=0.99, psi_local=0.43, a1=0.48):
    zlow = z < 2
    zhigh = z >= 3

    psi4375 = psi2 * np.ones_like(z)
    psi4375[zlow] = psi_local * (1 + z[zlow])**a1
    psi4375[zhigh] = psi3 * (1 + z[zhigh])**a2
   
    return psi4375


def fabs_V22(z, loglx, lognh, eps, psi2, beta=0.24, fctk=1):     
    nh2022 = np.logical_and(lognh >= 20, lognh < 22)
    nh2223 = np.logical_and(lognh >= 22, lognh < 23)
    nh2324 = np.logical_and(lognh >= 23, lognh < 24)
    nh2426 = np.logical_and(lognh >= 24, lognh <=26)
    
    f = np.zeros_like(z)
    f[nh2022] = (1 - _psi_V22(loglx[nh2022], z[nh2022], beta, psi2)) / 2
    f[nh2223] = (1 / (1 + eps)) * _psi_V22(loglx[nh2223], z[nh2223], beta, psi2)
    f[nh2324] = (eps / (1 + eps)) * _psi_V22(loglx[nh2324], z[nh2324], beta, psi2)
    f[nh2426] = fctk * _psi_V22(loglx[nh2426], z[nh2426], beta, psi2) / 2

    return f

def _psi_V22(loglx, z, beta, psi2, psi_min=0.2, psi_max=0.99):
    maxx = np.maximum(_psi43p75_V22(z, psi2) - beta*(loglx - 43.75), psi_min)
    psi = np.minimum(psi_max, maxx)

    return psi

def _psi43p75_V22(z, psi2, psi_local=0.43, a1=0.48):
    zlow = z < 2

    psi4375 = psi2 * np.ones_like(z)
    psi4375[zlow] = psi_local * (1 + z[zlow])**a1

    return psi4375


def _nonpar_fabs(z, loglx, lognh, *params):
    _, (z_m, loglx_m, lognh_m) = lumfun_nonpar_grid()
    xlaf_shape = (len(z_m), len(loglx_m), len(lognh_m))
    logphi = np.reshape(params, xlaf_shape)
    interp = RegularGridInterpolator((z_m, loglx_m, lognh_m), logphi, method="nearest", bounds_error=False, fill_value=None)

    return 10**interp(np.stack((z, loglx, lognh), axis=-1))


def _nonpar(z, loglx, logphi):
    _, (z_m, loglx_m, _) = lumfun_nonpar_grid()
    interp = RegularGridInterpolator((z_m, loglx_m), logphi, method="nearest", bounds_error=False, fill_value=None)

    return 10**interp(np.stack((z, loglx), axis=-1))
