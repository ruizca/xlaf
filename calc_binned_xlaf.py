"""
Calculation of binned X-ray luminosity function.

authors: E. Pouliasis & A. Ruiz
"""
from itertools import count
from pathlib import Path

import numpy as np
from rich.progress import track

import utils
from sampling import create_random_samples

data_path = Path(".", "data")


def binned_xlaf(
    sample,
    zlimits, 
    loglxlimits, 
    lognhlimits,
    nsamples=1000
):
    bins = _calc_bins(zlimits, loglxlimits, lognhlimits)
    rnd_samples = create_random_samples(sample, nsamples)
    xlaf, norm = None, None

    for i in range(nsamples):
        b, norm = _binned(rnd_samples[i, :, :], bins, norm=norm)

        if xlaf is None:
            xlaf = np.zeros((nsamples, *b.shape))
        
        xlaf[i, ...] = b

    # The area curves we are using go to zero for some flux values,
    # which translates in an null Omega for certain values of z, LX, NH.
    # This causes NaN/inf bins in the XLF. This usually happens in bins where
    # no sources are detected, son we change here non-finite values to zeros.
    # However, it is also possible for this to happen in populated bins (in which 
    # case their value is inf), so this correction can underestimate the XLF
    # for certain bins. For a more accurate estimation we need area curves
    # calculated considering the Poisson nature of the X-ray detection process,
    # where there is always a probability of detection for any flux, albeit small
    # (see Georgakakis+2008 for such procedure).
    mask_good = np.isfinite(xlaf)
    xlaf[~mask_good] = 0

    return xlaf, bins, rnd_samples


def _calc_bins(zlimits, loglxlimits, lognhlimits):
    n = int((zlimits[1] - zlimits[0]) / 0.25) + 1
    zbins = np.linspace(zlimits[0], zlimits[1], n)

    n = int((loglxlimits[1] - loglxlimits[0]) / 0.5) + 1
    # n = int((loglxlimits[1] - loglxlimits[0]) / 1.0) + 1
    loglxbins = np.linspace(loglxlimits[0], loglxlimits[1], n)
    # loglxbins = np.concatenate(([loglxbins[0]], loglxbins[2:]))

    if lognhlimits is not None:
        n = int((lognhlimits[1] - lognhlimits[0]) / 0.1) + 1
        lognhbins = np.linspace(lognhlimits[0], lognhlimits[1], n)

        return zbins, loglxbins, lognhbins
    
    else:
        return zbins, loglxbins


def _binned(sample, bins, norm=None):
    data = [sample[:, 0], sample[:, 1]]
    if len(bins) > 2:
        data += [sample[:, 2]]

    H, _ = np.histogramdd(data, bins=bins)

    if norm is None:
        norm = _calc_expected_number_of_sources(bins)

    return H / norm, norm


def _calc_expected_number_of_sources(bins):
    zbins = bins[0]
    loglxbins = bins[1]

    if len(bins) > 2:
        lognhbins = bins[2]
        with_lognh = True
    else:
        with_lognh = False

    omega_interp = utils.load_omega(data_path, with_lognh)

    if len(bins) > 2:
        grid_limits = (
            (zbins[0], zbins[-1]), 
            (loglxbins[0], loglxbins[-1]), 
            (lognhbins[0], lognhbins[-1]),
        )
        norm_shape = (len(zbins) - 1, len(loglxbins) - 1, len(lognhbins) - 1)
    else:
        grid_limits = (
            (zbins[0], zbins[-1]),
            (loglxbins[0], loglxbins[-1]),
        )
        norm_shape = (len(zbins) - 1, len(loglxbins) - 1)

    zgrid, loglxgrid, lognhgrid, omegagrid = utils.integration_grid(
        omega_interp, *grid_limits, num=100
    )
    norm = np.ones(norm_shape, dtype=float)

    for i, zmin, zmax in track(
        zip(count(), zbins[:-1], zbins[1:]), 
        description="Calculating expected sources for z/LX/NH bins"
    ):
        mask_z = np.logical_and(zgrid > zmin, zgrid <= zmax)

        for j, loglxmin, loglxmax in zip(count(), loglxbins[:-1], loglxbins[1:]):
            mask_loglx = np.logical_and(loglxgrid > loglxmin, loglxgrid <= loglxmax)
            mask = np.logical_and(mask_z, mask_loglx)
            
            if len(bins) > 2:
                for k, lognhmin, lognhmax in zip(count(), lognhbins[:-1], lognhbins[1:]):
                    mask_lognh = np.logical_and(lognhgrid > lognhmin, lognhgrid <= lognhmax)
                    mask2 = np.logical_and(mask, mask_lognh)   

                    norm[i, j, k] = np.sum(omegagrid[mask2])
            else:
                norm[i, j] = np.sum(omegagrid[mask])


    return norm


def main():
    sample_xxl_cosmos_cdf = utils.load_final_sample(data_path)
    integration_limits = utils.define_integration_limits()

    binned_xlaf(
        sample_xxl_cosmos_cdf,
        *integration_limits,
        nsamples=10000,
    )


if __name__ == "__main__":
    main()
