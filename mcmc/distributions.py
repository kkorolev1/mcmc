import jax
import jax.numpy as jnp
import jax.nn as nn
from jax import random
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)

class Distribution:
    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, key: random.PRNGKey, shape):   
        raise NotImplementedError 


class GaussianMixture(Distribution):
    def __init__(
        self,
        component_means: jnp.ndarray,
        component_covs: jnp.ndarray,
        mixin_coeffs=None,
    ):
        if mixin_coeffs is None:
            mixin_coeffs = jnp.full(
                (component_means.shape[0]),
                1 / component_means.shape[0],
                dtype=jnp.float32,
            )

        assert (
            component_means.shape[0] == component_covs.shape[0] == mixin_coeffs.shape[0]
        )
        assert jnp.allclose(sum(mixin_coeffs), 1)

        if component_means.ndim == 1:
            component_means = component_means[:, None]

        if component_covs.ndim == 1:
            component_covs = component_covs.reshape(component_means.shape[0], 1).repeat(
                component_means.shape[1], axis=1
            )

        if component_covs.ndim < 3:
            new_covs = jnp.zeros(
                (
                    component_means.shape[0],
                    component_means.shape[1],
                    component_means.shape[1],
                ),
                dtype=jnp.float32,
            )
            component_covs = fill_diagonal(new_covs, component_covs)

        self.component_means = component_means
        self.component_covs = component_covs
        self.mixin_coeffs = mixin_coeffs

    def log_prob(self, x):
        log_ps = multivariate_normal.logpdf(
            x, self.component_means, self.component_covs
        )
        return logsumexp(log_ps, b=self.mixin_coeffs)

    def sample(self, key: random.PRNGKey, shape):
        n_samples = shape[0]
        mixture_size = len(self.mixin_coeffs)
        key1, key2 = random.split(key)
        mixture_component = random.randint(key1, (n_samples,), 0, mixture_size)
        samples = random.multivariate_normal(
            key2,
            self.component_means,
            self.component_covs,
            shape=(n_samples, mixture_size),
        )
        return jnp.take_along_axis(
            samples, mixture_component[:, None, None], axis=1
        ).squeeze(axis=1)

    @property
    def mean(self):
        return jnp.dot(self.mixin_coeffs, self.component_means)

    @property
    def cov(self):
        mean = self.mean
        mean_var = (self.mixin_coeffs.reshape((-1, 1, 1)) * self.component_covs).sum(
            axis=0
        )
        var_mean = (
            self.mixin_coeffs.reshape((-1, 1, 1))
            * jnp.einsum(
                "ij,ik->ijk", self.component_means - mean, self.component_means - mean
            )
        ).sum(axis=0)
        return mean_var + var_mean

    @property
    def variance(self):
        return jnp.diag(self.cov)
    
class CauchyDistribution(Distribution):
    def __init__(self, dim):
        self.dim = dim

    def log_prob(self, x):
        c = 0.5 * (self.dim + 1)
        return jax.scipy.special.gammaln(c) - c * (jnp.log(jnp.pi) + jnp.log(x.T @ x + 1))

    def sample(self, key: random.PRNGKey, shape):
        n_samples = shape[0]
        return random.cauchy(key, shape=(n_samples, self.dim))
    
class BananaDistribution(Distribution):
    def __init__(self, dim, nu):
        self.dim = dim
        self.nu = nu

    def log_prob(self, x):
        ll = -1/self.nu * ((x[1::2] - x[::2]**2)**2) - (x[::2] - 1)**2
        return ll.sum(-1)

    def _transform(self, z):
        x = jnp.empty_like(z)
        x = x.at[:, ::2].set(1 + z[:, ::2] / jnp.sqrt(2))
        x = x.at[:, 1::2].set(x[:, ::2]**2 + jnp.sqrt(self.nu / 2) * z[:, 1::2])
        return x

    def sample(self, key: random.PRNGKey, shape):
        n_samples = shape[0]
        z = random.normal(key, shape=(n_samples, 2 * self.dim))
        return self._transform(z)
    

class LogisticRegressionPosterior(Distribution):
    def __init__(self, dim: int, sigma: float, X: jnp.ndarray, y: jnp.ndarray):
        self.dim = dim
        self.sigma = sigma
        self.X = X
        self.y = y

    def log_prob(self, theta):
        # theta in R^d
        log_prior = -0.5 * (theta**2).sum(axis=-1) / (self.sigma**2)
        log_likelihood = nn.log_sigmoid((self.X @ theta) * self.y).sum(axis=-1)
        return log_likelihood + log_prior