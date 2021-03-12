import numpy as np

import matplotlib.pyplot as plt
params = {'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern'}
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}', r'\usepackage{nicefrac}', r'\usepackage{bm}', r'\usepackage{upgreek}']
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams.update(params)

from scipy.integrate import solve_ivp

def sir(t, u, p):
    # --> Unpack parameters.
    β, γ = p

    # --> Unpack variables.
    s, i, r = u

    # --> Equations.
    ds = - β * s * i
    di = β * s * i - γ * i
    dr = γ * i

    return ds, di, dr

if __name__ == "__main__":

    # --> Paramètres de l'épidémie.
    β, γ = 0.1, 0.0333
    p = [β, γ]

    # --> Paramètres pour l'intégration.
    tspan = (0.0, 365.0)
    t_eval = np.linspace(tspan[0], tspan[1], 1024)
    u0 = np.array([0.999, 0.001, 0.0])

    # --> Simulate.
    sol = solve_ivp(
        lambda t, u : sir(t, u, p),
        tspan,
        u0,
        t_eval=t_eval
    )

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(6, 3/2))

    ax.plot(t_eval, sol.y[0], color="black", label="S")
    ax.plot(t_eval, sol.y[1], color="red", label="I", zorder=3)
    ax.plot(t_eval, sol.y[2], color="lightgray", label="R")

    ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.0), loc="lower center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set(yticks=[0, 1/3, 2/3, 1], yticklabels=[r"$0$", r"$\nicefrac{1}{3}$", r"$\nicefrac{2}{3}$", r"$1$"], ylim=(0, 1.02))
    ax.set(xlim=tspan, xlabel="Temps")

    plt.savefig("../imgs/sir_model.png", bbox_inches="tight", dpi=300)

    plt.show()
