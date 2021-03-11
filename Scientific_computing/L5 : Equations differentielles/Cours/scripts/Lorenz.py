import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t, u, p):
    # --> Unpack parameters.
    σ, ρ, β = p

    # --> Unpack state variables.
    x, y, z = u

    # --> Lorenz system.
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x*y - β * z

    return dx, dy, dz

def trial(u0, p, t_eval):

    tspan = (t_eval.min(), t_eval.max())

    sol = solve_ivp(
        lambda t, u : lorenz(t, u, p),
        tspan,
        u0,
        t_eval=t_eval,
        )

    return sol.y

def monte_carlo(u0, p, t_eval, scale=1e-4, ntrials=100):

    # --> Random perturbations.
    du = scale * np.random.randn(len(u0), ntrials)

    # --> Ensemble of initial conditions.
    u0 = u0.reshape(-1, 1) + du

    # --> Monte-Carlo simulations.
    u = np.array([trial(u0[:, i], p, t_eval) for i in range(ntrials)])

    return u

if __name__ == "__main__":

    # --> Paramètres du système.
    σ, ρ, β = 10.0, 28.0, 8/3
    p = [σ, ρ, β]

    # --> Paramètres pour l'intégration.
    tspan = (0.0, 40.0)
    t = np.linspace(tspan[0], tspan[1], 4096)
    u0 = np.array([1.0, 0.0, 0.0])

    # --> First pass to generate an initial condition on the attractor.
    sol = solve_ivp(
        lambda t, u: lorenz(t, u, p),
        tspan,
        u0,
        t_eval=t,
    )

    # --> Initial condition on the attractor.
    u0 = sol.y[:, -1]

    # -->
    u = monte_carlo(u0, p, t, scale=1e-4, ntrials=100)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    for x in u:
        ax.plot(x[0], x[-1], color="black", alpha=0.01)

    ax.set(xlim=(-20, 20), xticks=[])
    ax.set(ylim=(0, 50), yticks=[])

    ax.axis("off")
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)

    plt.savefig("../imgs/lorenz_attractor.png", bbox_inches="tight", dpi=300)

    #
    u_ref, u_mc = u[0], u[1:]

    for i in range(len(u_mc)):
        u_mc[i] -= u_ref

    erreur = np.array([np.linalg.norm(x, axis=0) for x in u_mc]).T
    erreur = (np.sqrt(u_ref.shape[1]) / np.linalg.norm(u_ref)) * erreur * 100

    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

    ax.semilogy(t, erreur, color="black", alpha=0.05)
    ax.semilogy(t, erreur.mean(axis=1), color="red")

    ax.set(xlim=(tspan[0], tspan[1]), xlabel="Temps")
    ax.set(ylim=(1e-4, 5e2), ylabel="Incertitude")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid("major", axis="y")

    plt.savefig("../imgs/lorenz_forecast_error.png", bbox_inches="tight", dpi=300)

    plt.show()
