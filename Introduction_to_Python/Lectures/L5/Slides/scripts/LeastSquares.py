import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.polynomial as npp

import matplotlib.pyplot as plt

def generate_data(n=50, σ=0.25, a=2, b=1):
    x = npr.uniform(low=0, high=2, size=n)
    y = a*x + b

    # --> Ajout du bruit.
    y += npr.normal(loc=0, scale=σ, size=n)

    return x, y

def main():

    # --> Generate data.
    x, y = generate_data()

    # --> Fit la droite.
    θ = npp.polynomial.polyfit(x, y, 1)
    print(θ)
    p = npp.Polynomial(θ)

    # --> Plot la figure.
    fig, ax = plt.subplots(1, 1, figsize=(9, 3))

    ax.scatter(x, y, color="black", alpha=0.5, label="Mesures")

    z = 1.2*np.linspace(0, 2, 128)
    ax.plot(z, p(z), color="red", label="Fit : $y = {0:.2f} + {1:.2f}x$".format(*θ))

    ax.set(xlim=(0, 2), ylim=(0, 6))
    ax.set(xlabel="$x$", ylabel="$y$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()

    plt.savefig("../imgs/least_squares_ex.png", bbox_inches="tight", dpi=600)

    plt.show()

if __name__ == "__main__":
    main()
