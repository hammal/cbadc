from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_state_dist(
    state_vectors: np.ndarray, filename: str, xlim: Tuple[float, float] = (-1, 1)
):
    # Estimate and plot densities using matplotlib tools.
    L_1_norm = np.linalg.norm(state_vectors, ord=1, axis=0)
    L_2_norm = np.linalg.norm(state_vectors, ord=2, axis=0)
    # Similarly, compute L_infty (largest absolute value) of the analog state
    # vector.
    L_infty_norm = np.linalg.norm(state_vectors, ord=np.inf, axis=0)

    bins = 100
    plt.rcParams["figure.figsize"] = [6.40, 4.80]
    fig, ax = plt.subplots(2)
    ax[0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    # ax[0].hist(L_1_norm, bins=bins, density=True, label='$p=1$')
    ax[0].hist(L_2_norm, bins=bins, density=True, label='$p=2$')
    ax[0].hist(
        L_infty_norm, bins=bins, density=True, color="orange", label='$p=\infty$'
    )
    plt.suptitle("Estimated probability densities")
    ax[0].set_xlabel("$\|\mathbf{x}(t)\|_p$")
    ax[0].set_ylabel("$p ( \| \mathbf{x}(t) \|_p ) $")
    ax[0].legend()
    ax[0].set_xlim((0, 1.5))
    for n in range(state_vectors.shape[0]):
        ax[1].hist(
            state_vectors[n, :],
            bins=bins,
            density=True,
            label="$x_{" + f'{n + 1}' + '}$',
        )
    ax[1].legend()
    ax[1].set_xlim(xlim)
    ax[1].set_xlabel("$x(t)_n$")
    ax[1].set_ylabel("$p ( x(t)_n )$")
    fig.tight_layout()
    fig.savefig(filename)
