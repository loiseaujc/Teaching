import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def kepler(t, u):
    # Unpack variables.
    x, dx, y, dy = u

    # Distance.
    r = np.sqrt(x**2 + y**2)

    # Equations du mouvement.
    ddx = -x / r**3
    ddy = -y / r**3

    return dx, ddx, dy, ddy

def moment_cinetique(x, dx, y, dy):
    J = x*dy - y*dx
    return J

if __name__ == "__main__":

    # --> Paramètres de la simulation.
    tspan = (0.0, 4.0)
    t = np.linspace(*tspan, 4096)
    u0 = np.array([1.0, 0.0, 0.0, 0.5])

    # --> Simulation.
    sol = solve_ivp(
        kepler,
        tspan,
        u0,
        t_eval=t,
        #method="DOP853",
        #rtol=1e-9
    )

    # --> Unpack variables.
    x, dx, y, dy = sol.y

    # --> Calcul du moment cinétique.
    J = moment_cinetique(x, dx, y, dy)

    # --> Plot solution.
    fig, ax = plt.subplots(1, 1)

    ax.plot(x, y, lw=1, color="lightgray")
    ax.plot(0, 0, "ro")

    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))

    # --> Vérification de la conservation du moment cinétique.
    fig, ax = plt.subplots(1, 1)

    ax.plot(t, J)

    ax.set(xlim=tspan, xlabel="Temps")
    ax.set(ylabel="Moment cinétique")

    plt.show()
