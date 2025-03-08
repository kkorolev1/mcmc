import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_samples(ax, name, dist, samples, alpha=0.1):
    samples = samples.reshape(-1, 2)
    x_bounds = samples[:, 0].min(), samples[:, 0].max()
    y_bounds = samples[:, 1].min(), samples[:, 1].max()
    x = jnp.linspace(x_bounds[0], x_bounds[1], 50)
    y = jnp.linspace(y_bounds[0], y_bounds[1], 50)
    X, Y = jnp.meshgrid(x, y)
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    #grid = jnp.concatenate((grid, jnp.zeros((grid.shape[0], dim - 2))), axis=-1)
    pdf = jnp.exp(jax.vmap(dist.log_prob)(grid)).reshape(X.shape)
    levels = jnp.linspace(pdf.min(), pdf.max(), 20)
    ax.contourf(X, Y, pdf, levels=levels, cmap='viridis', alpha=0.5)
    ax.scatter(samples[:, 0], samples[:, 1], s=5, c='green', alpha=alpha)
    ax.set(title=name, xlabel="x1", ylabel="x2")
        

def plot_all_samples(result_dict):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    method_names = sorted(result_dict["samples"].keys() - {"true"})
    for i, name in enumerate(method_names):
        samples = result_dict["samples"][name]
        plot_samples(axes[i], name, result_dict["dist"], samples)
    plt.tight_layout()
    plt.show()