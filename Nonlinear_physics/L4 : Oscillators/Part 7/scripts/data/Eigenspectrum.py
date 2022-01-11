import numpy as np

import matplotlib.pyplot as plt

def main():
    # --> Name of the file to load.
    fname = "Spectre_NSd_Re50.dat"

    # --> Load data.
    σ, ω, _ = np.loadtxt(fname, unpack=True)

    # --> Generate figure.
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.scatter(ω, σ, fc="none", ec="dodgerblue")

    ax.set(xlim=(-2, 2), xlabel=r"$\omega$")
    ax.set(ylim=(-0.4, 0.2), ylabel=r"$\sigma$")

    plt.show()

if __name__ == "__main__":
    main()
