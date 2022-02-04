import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def lorenz(t, u, p):
    # --> Unpack parameters.
    σ, ρ, β = p

    # --> Unpack state variables.
    x, y, z = u

    # --> Lorenz system.
    dx = σ*(y-x)
    dy = x*(ρ-z) - y
    dz = x*y - β*z

    return dx, dy, dz

def lorenz_attractor():
    # --> Parameters of the system.
    p = σ, ρ, β = 10, 28, 8/3

    # --> Parameters for the simulation.
    tspan = (0.0, 50.0)
    Δt = 0.01
    t = np.arange(tspan[1] // Δt) * Δt
    print(t.shape)

    # --> Run the simulation.
    u0 = np.random.randn(3)
    output = solve_ivp(
        lambda t, u : lorenz(t, u, p),
        tspan,
        u0
    )

    u0 = output["y"][:, -1]
    output = solve_ivp(
        lambda t, u : lorenz(t, u, p),
        tspan,
        u0,
        t_eval=t
    )

    data = np.c_[output["y"][0], output["y"][-1]]
    # np.savetxt("../data/lorenz_attractor.dat", data)

    # --> Points for the Lorenz map.
    def section(t, u):

        # --> Unpack parameters.
        σ, ρ, β = p

        # --> Unpack variables.
        x, y, z = u

        # --> Derivative.
        dz = x*y - β*z

        return dz

    section.direction = -1

    # --> Run the simulation.
    output = solve_ivp(
        lambda t, u: lorenz(t, u, p),
        (0, 1e3),
        u0,
        events=section
    )

    # --> Extract events.
    te, ze = output.t_events[0], output.y_events[0][:, -1]

    data = np.c_[ze[:-1], ze[1:]]
    np.savetxt("../data/lorenz_map.dat", data)

    plt.plot(data[:, 0], data[:, 1], ".")
    plt.show()


if __name__ == "__main__":
    lorenz_attractor()
