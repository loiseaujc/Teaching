import numpy as np

from scipy.linalg import expm

import matplotlib.pyplot as plt

def mass_spring(λ):
    A = np.array([[0.0, 1.0], [-1.0, -2*λ]])
    return A

def explicit_euler(x0, A, Δt, nsteps):
    x = np.zeros((nsteps, len(x0)))
    x[0] = x0
    for i in range(1, nsteps):
        x[i] = x[i-1] + Δt * A @ x[i-1]
    return x.T

def implicit_euler(x0, A, Δt, nsteps):
    x = np.zeros((nsteps, len(x0)))
    x[0] = x0
    H = np.eye(len(x0)) - Δt * A

    for i in range(1, nsteps):
        x[i] = np.linalg.solve(H, x[i-1])

    return x.T

def crank_nicholson(x0, A, Δt, nsteps):
    x = np.zeros((nsteps, len(x0)))
    x[0] = x0

    H1 = np.eye(len(x0)) - Δt/2 * A
    H2 = np.eye(len(x0)) + Δt/2 * A

    for i in range(1, nsteps):
        x[i] = np.linalg.solve(H1, H2 @ x[i-1])

    return x.T

if __name__ == "__main__":

    # --> Parameters of the model.
    λ = 0.1

    # --> Get the dynamics matrix.
    A = mass_spring(λ)

    # --> Analytic solution.
    t = np.linspace(0.0, 50.0, 1024)
    x0 = np.array([1.0, 0.0])

    x_analytique = np.array([expm(A*τ) @ x0 for τ in t]).T

    # --> Explicit Euler.
    Δt = 0.1
    nsteps = np.int(t[-1] // Δt)

    x_explicit = explicit_euler(x0, A, Δt, nsteps)
    x_implicit = implicit_euler(x0, A, Δt, nsteps)
    x_cn = crank_nicholson(x0, A, Δt, nsteps)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    ax.plot(t, x_analytique[0], color="black", label="Analytique")

    ax.set(xlim=(t[0], t[-1]), xlabel="Temps")
    ax.set(ylim=(-1.05, 1.05), ylabel="Position")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(ncol=2, bbox_to_anchor=(1, 1), loc="upper right")

    plt.savefig("../imgs/analytic_solution.png", bbox_inches="tight", dpi=300)

    ax.plot(Δt*np.arange(x_explicit.shape[1]), x_explicit[0], label="Euler explicite", color="dodgerblue")

    ax.legend(ncol=2, bbox_to_anchor=(1, 1), loc="upper right")

    plt.savefig("../imgs/explicit_solution.png", bbox_inches="tight", dpi=300)

    ax.plot(Δt*np.arange(x_implicit.shape[1]), x_implicit[0], label="Euler implicite", color="chartreuse")

    ax.legend(ncol=2, bbox_to_anchor=(1, 1), loc="upper right")

    plt.savefig("../imgs/implicit_solution.png", bbox_inches="tight", dpi=300)

    ax.plot(Δt*np.arange(x_cn.shape[1]), x_cn[0], label="Crank-Nicholson", ls="--", color="red")

    ax.legend(ncol=2, bbox_to_anchor=(1, 1), loc="upper right")

    plt.savefig("../imgs/crank_nicholson_solution.png", bbox_inches="tight", dpi=300)

    plt.show()
