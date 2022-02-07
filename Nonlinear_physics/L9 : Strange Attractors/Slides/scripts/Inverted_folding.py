import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def dynsys(t, u, a):
    # --> Unpack state variables.
    x, y, z = u

    # --> Rossler system
    dx = -y
    dy = x + z
    dz = x*z + a*y**2

    return dx, dy, dz

def dynsys_map():
    # --> Parameters for the simulation.
    a = 4

    # --> Parameters for the simulation.
    tspan = (0.0, 1e4)

    # --> Points for the Lorenz map.
    def section(t, u):
        # --> Unpack variables.
        x, y, z = u

        return y

    section.direction = -1

    # --> Rune the simulation.
    output = solve_ivp(
        lambda t, u : dynsys(t, u, a),
        tspan,
        np.random.randn(3),
        events=section
    )

    # --> Extract events.
    te, ze = output.t_events[0], output.y_events[0][100:, 0]

    data = np.c_[ze[:-1], ze[1:]]
    np.savetxt("../data/rossler_map.dat", data)

    plt.plot(data[:, 0], data[:, 1], ".")
    plt.show()
