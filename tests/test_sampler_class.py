import copy

import pytest

from zeus_bilby import Zeus


@pytest.fixture()
def SamplerClass():
    return Zeus


@pytest.fixture()
def create_sampler(SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    def create_fn(**kwargs):
        return SamplerClass(
            likelihood,
            priors,
            outdir=tmp_path / "outdir",
            label="test",
            use_ratio=False,
            **kwargs,
        )

    return create_fn


@pytest.fixture
def sampler(create_sampler):
    return create_sampler()


@pytest.fixture
def default_kwargs(sampler):
    return copy.deepcopy(sampler.default_kwargs)


def test_translate_kwargs(create_sampler):
    kwargs = dict(
        nwalkers=100,
        rstate0=[[1, 2], [3, 4]],
        lnprob0=[-1, -2],
    )
    sampler = create_sampler(**kwargs)
    translated = dict(
        nwalkers=100,
        rstate0=[[1, 2], [3, 4]],
        lnprob0=[-1, -2],
    )
    assert sampler._translate_kwargs(kwargs=translated) is None
    assert translated == dict(
        nwalkers=100,
        start=[[1, 2], [3, 4]],
        log_prob0=[-1, -2],
    )


def test_sampler_function_kwargs(sampler, default_kwargs):
    expected = {key: default_kwargs[key] for key in ("log_prob0", "start", "blobs0", "iterations", "thin")}
    assert sampler.sampler_function_kwargs == expected
