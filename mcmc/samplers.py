
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from numpyro.infer import MCMC, HMC
from numpyro.infer.mcmc import MCMCKernel
import typing as tp

from tqdm.notebook import trange

import optax
import paramax


def ula_step(prev_x, key: random.PRNGKey, params: dict):
    gamma = params["gamma"]
    grad_log_prob = params["grad_log_prob"]

    z = random.normal(key, shape=prev_x.shape)
    new_x = (
        prev_x
        + gamma * grad_log_prob(prev_x)
        + jnp.sqrt(2 * gamma) * z
    )
    return new_x, prev_x

def mala_step(prev_x, key: random.PRNGKey, params: dict):
    gamma = params["gamma"]
    log_prob = params["log_prob"]
    grad_log_prob = params["grad_log_prob"]

    step_key, proposal_key = jax.random.split(key, 2)
    new_x, _ = ula_step(prev_x, step_key, params)
    log_pi_diff = log_prob(new_x) - log_prob(prev_x)
    log_new_prev = (
        (new_x - prev_x - gamma * grad_log_prob(prev_x)) ** 2
    ).sum()
    log_prev_new = (
        (prev_x - new_x - gamma * grad_log_prob(new_x)) ** 2
    ).sum()
    p = jnp.exp(log_pi_diff + (-log_prev_new + log_new_prev) / (4 * gamma))
    u = jax.random.uniform(proposal_key)
    return jax.lax.cond(
        u <= p,
        lambda new_x, prev_x: (new_x, prev_x),
        lambda _, prev_x: (prev_x, prev_x),
        new_x,
        prev_x,
    )

def isir_step(prev_x, key: random.PRNGKey, params: dict):
    log_prob = params["log_prob"]
    n_proposals = params["n_proposals"]
    proposal_dist = params["proposal_dist"]
    key1, key2 = random.split(key)
    proposal_samples = proposal_dist.sample(key1, (n_proposals,))
    new_x = jnp.concat([prev_x[None, :], proposal_samples])
    log_weights = jax.vmap(log_prob)(new_x) - jax.vmap(proposal_dist.log_prob)(new_x)
    # jax.debug.print("{}", log_weights)
    # jax.debug.print("{}", log_weights.argmax())
    idx = random.categorical(key2, log_weights)
    return new_x[idx], prev_x


class Sampler:
    def __init__(
        self,
        dim: int,
        init_std: float = 1.0,
    ):
        self.dim = dim
        self.init_std = init_std

    def sample(
        self,
        key: jax.random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        n_chains: int = 1,
        skip_steps: int = 1,
    ):
        raise NotImplementedError


class LangevinSampler(Sampler):
    def __init__(
        self,
        step_fn: tp.Callable,
        params: dict,
        dim: int,
        init_std: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            init_std=init_std,
        )
        self.step_fn = step_fn
        self.params = params

    def step(self, prev_x, key: random.PRNGKey):
        return self.step_fn(prev_x, key)[0]

    @eqx.filter_jit
    def sample_chain(
        self,
        x: jnp.ndarray,
        key: random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        skip_steps: int = 1,
    ):
        keys = random.split(key, skip_steps * steps + burnin_steps)
        _, xs = jax.lax.scan(self.step_fn, init=x, xs=keys)
        xs = jnp.vstack(xs)
        return xs[burnin_steps:][::skip_steps]

    @eqx.filter_jit
    def sample(
        self,
        key: jax.random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        n_chains: int = 1,
        skip_steps: int = 1,
    ):
        key1, key2 = jax.random.split(key)
        starter_points = (
            jax.random.normal(key1, shape=(n_chains, self.dim)) * self.init_std
        )
        starter_keys = jax.random.split(key2, n_chains)
        samples = jax.vmap(self.sample_chain, in_axes=(0, 0, None, None, None))(
            starter_points, starter_keys, steps, burnin_steps, skip_steps
        )
        return samples.transpose(1, 0, 2)


class ULASampler(LangevinSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        init_std: float = 1.0,
    ):
        grad_log_prob = jax.jit(jax.grad(log_prob))
        params = {"log_prob": log_prob, "grad_log_prob": grad_log_prob, "gamma": gamma}
        step_fn = jax.tree_util.Partial(ula_step, params=params)

        super().__init__(
            step_fn=step_fn,
            params=params,
            dim=dim,
            init_std=init_std,
        )


class MALASampler(LangevinSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        init_std: float = 1.0,
    ):
        grad_log_prob = jax.jit(jax.grad(log_prob))
        params = {"log_prob": log_prob, "grad_log_prob": grad_log_prob, "gamma": gamma}
        step_fn = jax.tree_util.Partial(mala_step, params=params)
        super().__init__(
            step_fn=step_fn,
            params=params,
            dim=dim,
            init_std=init_std,
        )

class ISIRSampler(LangevinSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        proposal_dist,
        init_std: float = 1.0,
        n_proposals: int = 1,
    ):
        params = {"log_prob": log_prob, "proposal_dist": proposal_dist, "n_proposals": n_proposals}
        step_fn = jax.tree_util.Partial(isir_step, params=params)
        super().__init__(
            step_fn=step_fn,
            params=params,
            dim=dim,
            init_std=init_std,
        )

class PyroSampler(Sampler):
    def __init__(
        self,
        kernel: MCMCKernel,
        dim: int,
        init_std: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            init_std=init_std,
        )
        self.kernel = kernel

    def sample(
        self,
        key: jax.random.PRNGKey,
        steps: int = 1_000,
        burnin_steps: int = 1_000,
        n_chains: int = 1,
        skip_steps: int = 1,
    ):
        key1, key2 = jax.random.split(key)
        starter_points = (
            jax.random.normal(
                key1, shape=(n_chains, self.dim) if n_chains > 1 else (self.dim,)
            )
            * self.init_std
        )
        mcmc = MCMC(
            self.kernel,
            num_samples=steps,
            num_warmup=burnin_steps,
            num_chains=n_chains,
            thinning=skip_steps,
            jit_model_args=True,
            progress_bar=False,
            chain_method="parallel",
        )
        mcmc.run(key2, init_params=starter_points)
        samples = mcmc.get_samples(group_by_chain=True)
        return samples.transpose(1, 0, 2)


class HMCSampler(PyroSampler):
    def __init__(
        self,
        log_prob: tp.Callable,
        dim: int,
        gamma: float = 5e-3,
        init_std: float = 1.0,
    ):
        potential_fn = lambda x: -log_prob(x).squeeze()
        kernel = HMC(
            potential_fn=potential_fn,
            step_size=gamma,
            adapt_step_size=True,
            dense_mass=False,
        )
        super().__init__(
            kernel=kernel,
            dim=dim,
            init_std=init_std,
        )

class MaximumLikelihoodLoss:
    @eqx.filter_jit
    def __call__(self, params, static, x):
        dist = paramax.unwrap(eqx.combine(params, static))
        return -dist.log_prob(x).mean()

class AdaptiveNFSampler:
    def __init__(self, norm_flow, sampler: LangevinSampler, optimizer: optax.GradientTransformation, 
                 dim: int, lanvevin_per_resampling: int):
        self.norm_flow = norm_flow
        self.sampler = sampler
        self.optimizer = optimizer
        self.dim = dim
        self.lanvevin_per_resampling = lanvevin_per_resampling
        self.log_prob = self.sampler.params["log_prob"]

    @eqx.filter_jit
    def _step(self, params, *args, opt_state, loss, **kwargs):
        loss_value, grads = eqx.filter_value_and_grad(loss)(params, *args, **kwargs)
        updates, opt_state = self.optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value

    def resample(self, key: jax.random.PRNGKey, x: jnp.ndarray, **kwargs):
        raise NotImplementedError
    
    def sample(self, key: jax.random.PRNGKey, steps: int = 1_000, burnin_steps: int = 1_000, n_chains: int = 1, skip_steps: int = 1):
        params, static = eqx.partition(
            self.norm_flow,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        opt_state = self.optimizer.init(params)

        key, subkey = random.split(key)
        x = jax.random.normal(subkey, shape=(n_chains, self.dim))

        loss = MaximumLikelihoodLoss()
        loss_history = []
        pbar = trange(steps * skip_steps + burnin_steps)
        
        xs = []
        for iter in pbar:
            if iter % self.lanvevin_per_resampling == 0:
                key, subkey = random.split(key)
                x = self.resample(subkey, x)
            else:
                key, subkey = random.split(key)
                sampler_keys = random.split(subkey, x.shape[0])
                x = jax.vmap(self.sampler.step)(x, sampler_keys)
            xs.append(x)
            params, opt_state, loss_value = self._step(params, static, x, opt_state=opt_state, loss=loss)
            loss_history.append(loss_value)
            pbar.set_description_str(f"Loss: {loss_value:.3f}")
            norm_flow = eqx.combine(params, static)
        self.norm_flow = norm_flow
        self.loss_history = loss_history
        return jnp.stack(xs)[burnin_steps:][::skip_steps]
    
class IMH_NF_Sampler(AdaptiveNFSampler):
    def resample(self, key: jax.random.PRNGKey, x: jnp.ndarray):
        key1, key2 = random.split(key)
        new_x = self.norm_flow.sample(key1, (x.shape[0],))
        log_accept_prob = (jax.vmap(self.log_prob)(new_x) - jax.vmap(self.log_prob)(x)) - \
                          (jax.vmap(self.norm_flow.log_prob)(new_x) - jax.vmap(self.norm_flow.log_prob)(x))
        u = jax.random.uniform(key2, (x.shape[0],))
        accept_mask = u <= jnp.exp(log_accept_prob)
        x = x.at[accept_mask].set(new_x[accept_mask])
        return x
    
class ISIR_NF_Sampler(AdaptiveNFSampler):
    def __init__(self, norm_flow, sampler, optimizer, dim, lanvevin_per_resampling, n_proposals):
        super().__init__(norm_flow, sampler, optimizer, dim, lanvevin_per_resampling)
        self.n_proposals = n_proposals

    def resample(self, key: jax.random.PRNGKey, x: jnp.ndarray):
        params = {"proposal_dist": self.norm_flow, "log_prob": self.log_prob, "n_proposals": self.n_proposals}
        sampler_keys = random.split(key, x.shape[0])
        return jax.vmap(isir_step, in_axes=(0,0,None))(x, sampler_keys, params)[0]