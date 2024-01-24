from itertools import count

import numpy as np
from scipy.stats import norm as normal

from utils import lumfun_nonpar_grid


# width / low limit 
# Values from Vito+2014, with some adjustments
_default_priors = {
    "ldde": {
        "lognorm": [2, -5],  
        "loglstar": [4, 42],  
        "gamma1": [4, -2],
        "gamma2": [5, 1],
        "pden": [6, -10],
        "beta": [8, -3],
        # "zmin": [3, 2],
    }, 
    "pde": {
        "lognorm": [2, -5],  
        "loglstar": [4, 42],  
        "gamma1": [4, -2],
        "gamma2": [5, 1],
        "pden": [6, -10],
        # "zmin": [3, 2],
    }, 
    "lade": {
        "lognorm": [2, -5],  
        "loglstar": [4, 42],  
        "gamma1": [7, -2],
        "gamma2": [9, -4],
        "pden": [13, -9],
        "plum": [5, -2],
        # "zmin": [3, 2],
    },
    "fabs": {
        "logepsilon": [2, -1],
        "logfctk": [2.5, -1.5],
        "psi3": [100, 0],
        "c": [5, 0],
        "a2": [10, -10],
    }
}

def get_prior(model, fabs=True):
    if model == "nonpar":
        (z_bins, loglx_bins, lognh_bins), _ = lumfun_nonpar_grid()
        xlaf_shape = (len(z_bins), len(loglx_bins), len(lognh_bins))
        xlaf_size = np.prod(xlaf_shape)

        return lambda cube: _nonpar_prior2(cube, xlaf_shape), [f"logphi_{i}" for i in range(xlaf_size)]

    else:
        p = _default_priors[model]
        if fabs:
            p = {**p, **_default_priors["fabs"]}

        return lambda cube: _transform(cube, p.values()), list(p.keys())


def _transform(cube, values):
    params = cube.copy()

    for i, c, v in zip(count(), cube, values):
        params[i] = v[0]*c + v[1]

    return params


def _nonpar_prior(cube, shape):
    c = cube.reshape(shape)
    prior = np.zeros_like(c)

    log_phi_min, log_phi_max = -10, -1
    prior[0, 0, 0] = c[0, 0, 0] * (log_phi_max - log_phi_min) + log_phi_min

    sigma_loglx = 0.5
    for i in range(1, prior.shape[1]):
        prior[0, i, 0] = normal.ppf(c[0, i, 0], loc=prior[0, i - 1, 0], scale=sigma_loglx)

    sigma_z = 0.5
    for i in range(1, prior.shape[0]):
        prior[i, :, 0] = normal.ppf(c[i, :, 0], loc=prior[i - 1, :, 0], scale=sigma_z)

    sigma_lognh = 0.75
    for i in range(1, prior.shape[2]):
        prior[:, :, i] = normal.ppf(c[:, :, i], loc=prior[:, :, i - 1], scale=sigma_lognh)

    return prior.flatten()


def _nonpar_prior2(cube, shape):
    c = cube.reshape(shape)

    log_phi_min, log_phi_max = -10, -1
    sigma_loglx, sigma_z, sigma_lognh = 0.5, 0.5, 0.75

    prior = np.zeros_like(c)
    # prior[:2, :2, 0] = c[:2, :2, 0] * (log_phi_max - log_phi_min) + log_phi_min
    prior[0, 0, 0] = c[0, 0, 0] * (log_phi_max - log_phi_min) + log_phi_min
    prior[0, 1, 0] = normal.ppf(c[0, 1, 0], loc=prior[0, 0, 0], scale=sigma_loglx)
    prior[1, :2, 0] = normal.ppf(c[1, :2, 0], loc=prior[0, :2, 0], scale=sigma_z)

    # Constant slope logLX
    for i in range(2, prior.shape[1]):
        loc = prior[:2, i - 1, 0] - prior[:2, i - 2, 0]
        prior[:2, i, 0] = prior[:2, i - 1, 0] + normal.ppf(c[:2, i, 0], loc=loc, scale=sigma_loglx)

    # Constant slope z
    for i in range(2, prior.shape[0]):
        loc = prior[i - 1, :, 0] - prior[i - 2, :, 0]
        prior[i, :, 0] = prior[i - 1, :, 0] + normal.ppf(c[i, :, 0], loc=loc, scale=sigma_z)

    # Constant value logNH
    for i in range(1, prior.shape[2]):
        prior[:, :, i] = normal.ppf(c[:, :, i], loc=prior[:, :, i - 1], scale=sigma_lognh)

    return prior.flatten()
