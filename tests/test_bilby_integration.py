"""Test the integration with bilby"""

import bilby


def test_sampling_zeus(
    bilby_gaussian_likelihood_and_priors,
    tmp_path,
    n_pool,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    outdir = tmp_path / "test_sampling_zeus"

    bilby.run_sampler(
        outdir=outdir,
        resume=False,
        plot=False,
        likelihood=likelihood,
        priors=priors,
        sampler="zeus",
        n_walkers=10,
        n_steps=10,
        injection_parameters={"x": 0.0, "y": 0.0},
        n_pool=n_pool,
    )
