import bilby
import numpy as np
import pytest


class GaussianLikelihood(bilby.Likelihood):
    def __init__(self):
        super().__init__(parameters={"x": None, "y": None})

    def log_likelihood(self, parameters=None):
        return -0.5 * (parameters["x"] ** 2 + parameters["y"] ** 2) - np.log(2.0 * np.pi)


@pytest.fixture
def bilby_gaussian_likelihood_and_priors():
    likelihood = GaussianLikelihood()
    priors = dict(
        x=bilby.core.prior.Uniform(-10, 10, "x"),
        y=bilby.core.prior.Uniform(-10, 10, "y"),
    )
    return likelihood, priors


@pytest.fixture(params=[1, 2])
def n_pool(request):
    return request.param
