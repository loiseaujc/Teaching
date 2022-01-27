import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d

φ = lambda x : -np.tanh(x)
X = lambda t, x : x + t * φ(x)

if __name__ == "__main__":

    nx, nt = 1024, 256

    x0 = np.linspace(-10, 10, nx)
    x = np.array([X(τ, x0) for τ in  np.linspace(0, 1, nt)])
    t = np.linspace(0, 20, nt).reshape(-1, 1) @ np.ones((1, nx))
    Z = np.array([φ(x0) for i in range(nt)])

    characteristics = np.stack((x, t), axis=2).transpose(1, 0, 2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    ax.scatter(x, t, marker=".", color="white", s=0)
    ax.add_collection(LineCollection(characteristics, color="k", lw=1, alpha=0.5))

    ax.set(xlim=(-1, 1), xlabel=r"$x$", xticks=[])
    ax.set(ylim=(t.min(), t.max()), ylabel=r"$t$", yticks=[])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plt.savefig("imgs/burgers_characteristics.png", bbox_inches="tight", dpi=1200)

    plt.show()
