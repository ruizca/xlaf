from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
from scipy.interpolate import RegularGridInterpolator

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)


def define_integration_limits():
    # This defines the integration limits used for calculating the LF
    zlimits = (3, 6)
    loglxlimits = (42, 47)  #42.2-46.7 n  cosmos"43.40,46.30  #all:42.2,46.7
    lognhlimits = (20, 26)  #20.5, 25.3   #cosmos: 21-25.1, 43.40-46.30

    return zlimits, loglxlimits, lognhlimits


def load_final_sample(data_path: Path) -> Table:
    samples_path = data_path / "samples"
    xxln = Table.read(samples_path / "xray_sample_xxln_ids.fits")
    ccls = Table.read(samples_path / "xray_sample_ccls_ids.fits")
    cdf = Table.read(samples_path / "xray_sample_cdf_ids.fits")

    return vstack([xxln, ccls, cdf], metadata_conflicts="silent")


def integration_grid(omega, zlimits, loglxlimits, lognhlimits=None, num=100):
    z = np.linspace(*zlimits, num=num)
    dz = np.diff(z)

    loglx = np.linspace(*loglxlimits, num=num)
    dloglx = np.diff(loglx)

    diff = dz[0] * dloglx[0]

    if lognhlimits is not None:
        lognh = np.linspace(*lognhlimits, num=num)
        dlognh = np.diff(lognh)        
        
        diff *= dlognh[0]

        zgrid, loglxgrid, lognhgrid = np.meshgrid(z, loglx, lognh, indexing="ij")
        grid = np.stack((zgrid, loglxgrid, lognhgrid), axis=-1)

    else:
        zgrid, loglxgrid = np.meshgrid(z, loglx, indexing="ij")
        lognhgrid = None
        grid = np.stack((zgrid, loglxgrid), axis=-1)

    Omega = omega(grid)
    dvdz_diff = diff * cosmo.differential_comoving_volume(zgrid).value

    return zgrid, loglxgrid, lognhgrid, Omega * dvdz_diff


def load_omega(data_path: Path, with_lognh=True) -> RegularGridInterpolator:
    # This function return a fast grid interpolator for the total effective area
    # of the survey as a function of redshift, luminosity and NH. It was
    # calculated following Laloux+2023, assuming a UXClumpy model
    npzfile = np.load(data_path / "omega_total_sr_interpolation_grid.npz")
    z = npzfile["z"]
    loglx = npzfile["loglx"]
    lognh = npzfile["lognh"]
    omega = npzfile["omega"]  # Total effective area of the survey in steradians

    if with_lognh:
        omega_interpolator = RegularGridInterpolator((z, loglx, lognh), omega)
    else:
        omega = np.trapz(omega, lognh, axis=-1) / np.trapz(np.ones(len(lognh)), lognh)
        omega_interpolator = RegularGridInterpolator((z, loglx), omega)

    return omega_interpolator


def lumfun_nonpar_grid():
    z_edges = np.linspace(3, 6, num=4)
    z_bins = [(min, max) for min, max in zip(z_edges[:-1], z_edges[1:])]
    z_m = (z_edges[:-1] + z_edges[1:]) / 2

    loglx_edges = np.linspace(42, 47, num=11)
    # loglx_edges = np.linspace(42, 47, num=6)
    loglx_bins = [(min, max) for min, max in zip(loglx_edges[:-1], loglx_edges[1:])]
    loglx_m = (loglx_edges[:-1] + loglx_edges[1:]) / 2

    lognh_edges = np.array([20, 23, 24, 26])
    lognh_bins = [(min, max) for min, max in zip(lognh_edges[:-1], lognh_edges[1:])]
    lognh_m = (lognh_edges[:-1] + lognh_edges[1:]) / 2

    return (z_bins, loglx_bins, lognh_bins), (z_m, loglx_m, lognh_m)
