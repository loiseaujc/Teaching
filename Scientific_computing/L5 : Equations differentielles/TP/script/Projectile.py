import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

sec = lambda θ : 1/np.cos(θ)

def impact(t, u):
    x, dx, y, dy = u
    return y
impact.terminal = True
impact.direction = -1

def sans_frottement(t, u):
    # Unpack variables.
    x, dx, y, dy = u

    # Equations du mouvement.
    ddx = 0
    ddy = -1

    return dx, ddx, dy, ddy

def avec_frottement(t, u, α=0.1):

    # Unpack variable
    x, dx, y, dy = u

    # Equations du mouvements.
    ddx = -α*dx
    ddy = -α*dy - 1

    return dx, ddx, dy, ddy

def trajectoire_parabolique(x, θ):
    y = -0.5 * sec(θ)**2 * x**2 + np.tan(θ) * x
    return y

def trajectoire_avec_frottement(x, θ, α=0.1):
    y = (np.tan(θ) + sec(θ)/α) * x + np.log(1 - α*sec(θ)*x) / α**2
    return y

if __name__ == "__main__":

    # -->
    θ = np.pi/4

    tspan = (0.0, 2.0)

    t = np.linspace(*tspan, 1024)

    x0 = np.array([0.0, np.cos(θ), 0.0, np.sin(θ)])

    sol_sans_frottement = solve_ivp(sans_frottement, tspan, x0, t_eval=t, events=impact)
    sol_avec_frottement = solve_ivp(avec_frottement, tspan, x0, t_eval=t, events=impact)

    #####

    x = np.linspace(0, 1, 1024)

    fig, ax = plt.subplots(1, 1)

    ax.plot(
        sol_sans_frottement.y[0], sol_sans_frottement.y[2],
        color="black",
        label="Sans frottement"
        )

    ax.plot(
        x, trajectoire_parabolique(x, θ),
        color="red",
        label="Analytique (sans frottement)",
        ls="--"
    )

    ax.plot(
        sol_avec_frottement.y[0], sol_avec_frottement.y[2],
        color="gray",
        label="Avec frottement",
        )

    ax.plot(
        x, trajectoire_avec_frottement(x, θ),
        color="dodgerblue",
        ls="--",
        label="Analytique (sans frottement)"
        )

    ax.set(ylim=(0, 0.5), ylabel="Altitude")
    ax.set(xlim=(0, 1.01), xlabel="Distance")

    ax.set(aspect="equal")

    ax.legend()

    plt.show()
