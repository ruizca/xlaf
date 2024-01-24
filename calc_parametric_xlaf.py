"""
Calculation of parametric X-ray luminosity and absorption functions.

authors: E. Pouliasis & A. Ruiz
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from ultranest import ReactiveNestedSampler, stepsampler
from ultranest.mlfriends import RobustEllipsoidRegion

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

except ImportError:
    rank = 0

import utils
from xlf_models import xlf, fabs as funabs
from xlf_priors import get_prior
from sampling import create_ims_sample


data_path = Path(".", "data")


class XLF_loglikelihood:
    def __init__(self, zlimits, loglxlimits, lognhlimits, ims, ims_weights, model="pde"):
        self.fabs = self._set_fabs(lognhlimits)    
        self.lumfun = self._set_lumfun(model)
        self.omega = self._set_omega()
        self.dvdz = self._set_diff_comoving_volume()
        self._integration_grid = self._set_integration_grid(
            zlimits, loglxlimits, lognhlimits, num=50
        )
        self.ims = ims
        self.ims_weights = ims_weights

        # This is needed for the calculation of the likelihood
        # See Buchner+2015 for an explanation of why the area curve (Omega) is used here.
        self.Omega_ims_dvdz_ims = self.omega(self.ims) * self.dvdz(self.ims[:, :, 0])
            
    @property
    def zgrid(self):
        return self._integration_grid[0]

    @property
    def loglxgrid(self):
        return self._integration_grid[1]

    @property
    def lognhgrid(self):
        return self._integration_grid[2]

    @property
    def omegagrid(self):
        return self._integration_grid[-1] 

    def _set_fabs(self, lognhlimits):
        if lognhlimits is None:
            fabs = False
        else:
            fabs = True
        
        return fabs
    
    def _set_lumfun(self, model):
        return xlf(model, fabs=self.fabs)

    def _set_omega(self):
        return utils.load_omega(data_path, with_lognh=self.fabs)

    def _set_diff_comoving_volume(self):
        # Calculations of differential_comoving_volume are quite slow
        # We define an interpolator to speed up the likelihood calculation
        z = np.linspace(0.01, 10, num=1000)
        dvdz = utils.cosmo.differential_comoving_volume(z).value  # per sr

        return interp1d(z, dvdz)

    def _set_integration_grid(self, zlimits, loglxlimits, lognhlimits, num=100):
        return utils.integration_grid(self.omega, zlimits, loglxlimits, lognhlimits, num)
    
    def log_likelihood(self, params):
        # Likelihood from Aird+10, Buchner+15
        # It assumes a Poisson process and it is the probability of observing
        # the sources in the sample times the probability of not observing any other source (Loredo 2004)
        logL = self._observed_number_of_sources(params) - self._expected_number_of_sources(params) 
        return logL

    def _expected_number_of_sources(self, params):
        coords = [self.zgrid, self.loglxgrid]
        if self.fabs:
            coords.append(self.lognhgrid)
        #     renorm = 1 / np.trapz(funabs(*coords, *params[-5:]), self.lognhgrid[0, 0, :], axis=-1)
        # else:
        #     renorm = 1

        phi = self.lumfun(*coords, *params)

        # if self.fabs:
        #     mask_lognh_ctk = self.lognhgrid >= 24
        #     phi[mask_lognh_ctk] = 0

        return np.sum(phi * self.omegagrid)

    def _observed_number_of_sources(self, params):
        coords = [self.ims[:, :, 0], self.ims[:, :, 1]]  # z, loglx
        if self.fabs:
            coords.append(self.ims[:, :, 2])  # lognh

        # /int_grid p(z, LX, NH) * phi(z, LX, NH)
        # We solve this integral via importance sampling
        # (although not really, because this is not importance sampling: 
        # for each source we sampled only from the region of interest and then applied a weight 
        # corresponding to the integral of p(z, LX, NH) in that region)
        integrals = self.ims_weights * np.mean(
            self.lumfun(*coords, *params) * self.Omega_ims_dvdz_ims, axis=1
        ) 

        return np.sum(np.log(integrals))

    # def _ons_with_mci(params):
    #     # /int_grid p(z, LX, NH) * phi(z, LX, NH)
    #     # We solve the integral via montecarlo integration:
    #     # we sample from the whole posterior and keep only the 
    #     # realizations within the region of interest. The integral
    #     # is \sum_grid phi(z, LX, NH) / nsamples
    #     integrals = [
    #         np.sum(
    #             lumfun(src["inlimits"][0, :], src["inlimits"][1, :], src["inlimits"][2, :], *params)
    #             * sky_area_total_sr
    #             * omega_nh_3d(np.stack((src["inlimits"][0, :], src["inlimits"][1, :], src["inlimits"][2, :]), axis=-1))
    #             * cosmo.differential_comoving_volume(src["inlimits"][0, :]).value
    #         ) / src["nsamples"]
    #         for src in mci
    #     ]

    #     return integrals


def fitmodel(
    sample,
    zlimits, 
    loglxlimits, 
    lognhlimits=None,
    model="ldde",
    run_ultranest=True,
    use_stepsampler=False,
    log_dir=".",
    **kwargs,
):
    try:
        ims_sample = np.load(data_path / "ims_sample3.npz")
        ims = ims_sample["ims"]
        ims_weights = ims_sample["ims_weights"]
        ultranest_resume = "resume"

        if rank == 0:
            print("Existing sample for MC integration loaded!")

    except FileNotFoundError:
        ims, ims_weights = create_ims_sample(sample, zlimits, loglxlimits, lognhlimits, nsamples=1000)
        np.savez(data_path / "ims_sample3.npz", ims=ims, ims_weights=ims_weights)
        ultranest_resume = "overwrite"

    xlf_logl = XLF_loglikelihood(zlimits, loglxlimits, lognhlimits, ims, ims_weights, model)
    prior, parameters = get_prior(model, fabs=(lognhlimits is not None))

    if run_ultranest:
        sampler = ReactiveNestedSampler(
            parameters,
            xlf_logl.log_likelihood,
            prior,
            log_dir=log_dir,
            resume=ultranest_resume,
        )
        if use_stepsampler:
            sampler.stepsampler = stepsampler.SliceSampler(
                nsteps=len(parameters),
                generate_direction=stepsampler.generate_mixture_random_direction,
                # adaptive_nsteps=False,
                # max_nsteps=400
            )
        sampler.run(min_num_live_points=400, frac_remain=0.01, **kwargs)

        if rank == 0:
            sampler.print_results()
            # sampler.plot()
    

def main():
    sample_xxl_cosmos_cdf = utils.load_final_sample(data_path)
    integration_limits = utils.define_integration_limits()
    model = "nonpar"

    fitmodel(
        sample_xxl_cosmos_cdf,
        # *integration_limits[:2],
        *integration_limits,
        model=model,
        run_ultranest=True,
        use_stepsampler=True,
        log_dir=f"./data/ultranest/{model}_xlaf3_test",
        region_class=RobustEllipsoidRegion,
        dlogz=0.5 + 0.1 * 90,
        update_interval_volume_fraction=0.4,
        max_num_improvement_loops=3,
    )


if __name__ == "__main__":
    main()
