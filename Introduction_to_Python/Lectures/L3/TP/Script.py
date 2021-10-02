import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

def generate_data(ntrials=100, nsteps=200):
    # --> Generate white noise process.
    x = npr.randn(nsteps, ntrials)
    x = x.cumsum(axis=0)
    return x

def main(ntrials=200, nsteps=400):

    # -->
    x = generate_data(ntrials=ntrials, nsteps=nsteps+1)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(9, 3))

    ax.plot(
        x,
        color="black",
        alpha=0.1)

    ax.set(xlim=(0, nsteps), xlabel="Temps")
    ax.set(ylim=(-50, 50), ylabel="Déplacement")

    plt.savefig("figure_1.png", bbox_inches="tight", dpi=300)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    for i in [0, 4, 9, 19, 39]:
        ax.hist(
            x[i],
            density=True,
            alpha=0.5,
            label="T = {0}".format(i+1)
            )

    ax.set(xlim=(-25, 25), xlabel="Déplacement")
    ax.set(ylim=(0, 0.5), ylabel="Probabilité")

    ax.legend(loc="best")

    plt.savefig("figure_2.png", bbox_inches="tight", dpi=300)

    plt.show()

if __name__ == "__main__":

    main()
