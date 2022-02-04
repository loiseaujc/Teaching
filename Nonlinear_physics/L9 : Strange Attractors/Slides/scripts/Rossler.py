import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def rossler(t, u, p):
    # --> Unpack parameters.
    a, b, c = p

    # --> Unpack state variables.
    x, y, z = u

    # --> Rossler system
    dx = -y-z
    dy = x + a*y
    dz = b + z*(x-c)

    return dx, dy, dz

def rossler_map():
    # --> Parameters for the simulation.
    p = a, b, c = 0.2, 0.2, 5.7

    # --> Parameters for the simulation.
    tspan = (0.0, 1e4)

    # --> Points for the Lorenz map.
    def section(t, u):
        # --> Unpack variables.
        x, y, z = u

        xp = (c - np.sqrt(c**2-4*a*b)) / (2*a)

        return x - xp

    section.direction = -1

    # --> Rune the simulation.
    output = solve_ivp(
        lambda t, u : rossler(t, u, p),
        tspan,
        np.random.randn(3),
        events=section
    )

    # --> Extract events.
    te, ze = output.t_events[0], output.y_events[0][100:, 1]

    data = np.c_[ze[:-1], ze[1:]]
    np.savetxt("../data/rossler_map.dat", data)

    plt.plot(data[:, 0], data[:, 1], ".")
    plt.show()

def period_doubling():

    cs = 4, 6, 8.5, 8.7
    tspan = (0.0, 1e4)
    Δt = 0.25
    t = np.arange(tspan[1]//Δt) * Δt

    for (i, c) in enumerate(cs):
        p = 0.1, 0.1, c

        # --> Run the simulation.
        output = solve_ivp(
            lambda t, u : rossler(t, u, p),
            tspan,
            np.random.randn(3),
            t_eval=t,
        )

        x, y = output["y"][0], output["y"][1]
        x, y = x[t>9500], y[t>9500]

        data = np.c_[x, y]
        np.savetxt("../data/rossler_orbit_{0}.dat".format(i+1), data)


if __name__ == "__main__":
    period_doubling()
