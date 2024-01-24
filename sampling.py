from pathlib import Path

import numpy as np
from rich.progress import track

rng = np.random.default_rng()
POSTERIOR_PATH = Path(".", "data", "posteriors")


def _load_posterior(srcid):
    pdf_path = POSTERIOR_PATH / f"{srcid}_posterior.dat"
    return np.loadtxt(pdf_path, skiprows=1, unpack=True)  # lognh, gamma, z, logfx, loglx


def create_ims_sample_old(sample, zlimits, loglxlimits, lognhlimits, nsamples=100000):
    ims_sample = []
    for srcid in track(sample["srcid"]):
        lognh, _, z, _, loglx = _load_posterior(srcid)

        mask_z = np.logical_and(z >= zlimits[0], z <= zlimits[1])
        mask_loglx = np.logical_and(loglx >= loglxlimits[0], loglx <= loglxlimits[1])
        mask_lognh = np.logical_and(lognh >= lognhlimits[0], lognh <= lognhlimits[1])

        mask = np.logical_and(mask_z, mask_loglx)
        mask = np.logical_and(mask, mask_lognh)

        # If this check fails, it means that we are using sources that shouldn't be
        # in the sample, or that the integration limits are wrong
        assert len(z[mask]) > 0

        idx = rng.integers(len(z[mask]), size=nsamples)
        ims_sample.append(
            {
                "weight": len(z[mask]) / len(z),
                "inlimits": np.array([z[mask][idx], loglx[mask][idx], lognh[mask][idx]]),
            }
        )        

    return ims_sample


def create_ims_sample(sample, zlimits, loglxlimits, lognhlimits=None, nsamples=100000):
    if lognhlimits is None:
        npars = 2
    else:
        npars = 3

    ims_sample = np.zeros((len(sample), nsamples, npars))
    weights = np.zeros(len(sample))

    for i, srcid in enumerate(track(sample["srcid"], description="Sampling for MC integration")):
        lognh, _, z, _, loglx = _load_posterior(srcid)

        mask_z = np.logical_and(z >= zlimits[0], z <= zlimits[1])
        mask_loglx = np.logical_and(loglx >= loglxlimits[0], loglx <= loglxlimits[1])
        mask = np.logical_and(mask_z, mask_loglx)

        if lognhlimits is not None:
            mask_lognh = np.logical_and(lognh >= lognhlimits[0], lognh <= lognhlimits[1])
            mask = np.logical_and(mask, mask_lognh)

        # If this check fails, it means that we are using sources that shouldn't be
        # in the sample, or that the integration limits are wrong
        assert len(z[mask]) > 0

        idx = rng.integers(len(z[mask]), size=nsamples)
        weights[i] = len(z[mask]) / len(z)

        ims_sample[i, :, 0] = z[mask][idx]
        ims_sample[i, :, 1] = loglx[mask][idx]
        if lognhlimits is not None:
            ims_sample[i, :, 2] = lognh[mask][idx]
        

    return ims_sample, weights


def create_mci_sample(sample, zlimits, loglxlimits, lognhlimits, nsamples=100000):
    mci_sample = []
    for srcid in track(sample["srcid"]):
        lognh, _, z, _, loglx = _load_posterior(srcid)

        idx = rng.integers(len(z), size=nsamples)

        mask_z = np.logical_and(z[idx] >= zlimits[0], z[idx] <= zlimits[1])
        mask_loglx = np.logical_and(loglx[idx] >= loglxlimits[0], loglx[idx] <= loglxlimits[1])
        mask_lognh = np.logical_and(lognh[idx] >= lognhlimits[0], lognh[idx] <= lognhlimits[1])

        mask = np.logical_and(mask_z, mask_loglx)
        mask = np.logical_and(mask, mask_lognh)

        # If this check fails, it means that we are using sources that shouldn't be
        # in the sample, or that the integration limits are wrong
        assert len(z[idx][mask]) > 0

        mci_sample.append(
            {
                "nsamples": nsamples,
                "inlimits": np.array([z[idx][mask], loglx[idx][mask], lognh[idx][mask]]),
            }
        )

    return mci_sample


def create_random_samples(sample, nsamples=100000):
    rnd_sample = np.zeros((nsamples, len(sample), 3))

    for i, srcid in enumerate(track(sample["srcid"], description="Sampling from posteriors")):
        lognh, _, z, _, loglx = _load_posterior(srcid)

        idx = rng.integers(len(z), size=nsamples)
        rnd_sample[:, i, 0] = z[idx]
        rnd_sample[:, i, 1] = loglx[idx]
        rnd_sample[:, i, 2] = lognh[idx]

    return rnd_sample
