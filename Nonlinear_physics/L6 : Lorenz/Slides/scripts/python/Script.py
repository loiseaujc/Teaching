import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

def lorenz(t, u, p):

    # --> Unpack parameters.
    σ, ρ, β = p

    # --> Unpack state variables.
    x, y, z = u

    # --> Lorenz system.
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x*y - β*z

    return dx, dy, dz

def transient_chaos():
    # --> Parameters.
    σ, ρ, β = 10, 22, 8/3
    p = [σ, ρ, β]

    # --> Setup the simulation.
    tspan = (0.0, 1)
    Δt = 0.05
    t = np.arange(0, tspan[1]//Δt) * Δt
    # u0 = 50*np.random.randn(3)
    u0 = [50, 50, 50]

    # --> Run the simulation.
    output = solve_ivp(
        lambda t, u: lorenz(t, u, p),
        tspan,
        u0,
        t_eval=t,
    )

    u0 = output["y"][:, -1]

    # --> Run the simulation.
    tspan = (0.0, 50.0)
    Δt = 0.1

    for i in range(3):
        t = np.arange(0, tspan[1]//Δt) * Δt

        output = solve_ivp(
            lambda t, u: lorenz(t, u, p),
            tspan,
            u0 + 1e-1*np.random.randn(3),
            t_eval=t,
        )

        t, x = output["t"], output["y"][0]
        t, x = t[t<50], x[t<50]
        data = np.c_[t, x]
        plt.plot(output["t"], output["y"][0])
        np.savetxt("../../data/lorenz_transients_trial_{0}.dat".format(i+1), data)
    plt.show()

def strange_attractor():
    # --> Parameters.
    σ, ρ, β = 10, 28, 8/3
    p = [σ, ρ, β]

    # --> Setup the simulation.
    tspan = (0.0, 100.0)
    Δt = 0.01
    t = np.arange(0, tspan[1]//Δt) * Δt
    u0 = np.array([50, 0, 50])

    # --> Run the simulation.
    output = solve_ivp(
        lambda t, u: lorenz(t, u, p),
        tspan,
        u0,
        t_eval=t,
    )

    data = output["y"][:, output["t"] > 50].T
    data[:, 0] += 25

    plt.plot(data[:, 0], data[:, 2])
    plt.show()
    np.savetxt("../../data/lorenz_attractor.dat", data)

def poincare_section():
    # --> Parameters.
    σ, ρ, β = 10, 28, 8/3
    p = [σ, ρ, β]

    # --> Setup the simulation.
    tspan = (0.0, 1000.0)
    Δt = 0.01
    t = np.arange(0, tspan[1]//Δt) * Δt
    u0 = np.array([50, 0, 50])

    # --> Points for the Lorenz map.
    def section(t, u):

        # --> Unpack parameters.
        σ, ρ, β = p

        # --> Unpack variables.
        x, y, z = u

        return z - (ρ - 1)

    # --> Run the simulation.
    output = solve_ivp(
        lambda t, u: lorenz(t, u, p),
        tspan,
        u0,
        t_eval=t,
        events=section
    )

    # --> Extract events.
    te, ue = output.t_events[0], output.y_events[0]
    te, ue = te[te>50], ue[te>50]
    data = np.c_[ ue[:, 0], ue[:, 1]]
    np.savetxt("../../data/poincare_section.dat", data)

def fractal_basin():
    pass

def lorenz_map():
    # --> Parameters.
    σ, ρ, β = 10, 28, 8/3
    p = [σ, ρ, β]

    # --> Setup the simulation.
    tspan = (0.0, 200.0)
    Δt = 0.01
    t = np.arange(0, tspan[1]//Δt) * Δt
    u0 = np.array([50, 0, 50])

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
        tspan,
        u0,
        t_eval=t,
        events=section
    )

    # --> Extract events.
    te, ze = output.t_events[0], output.y_events[0][:, -1]

    data = np.c_[ze[:-1], ze[1:]]
    np.savetxt("../../data/lorenz_map.dat", data)

    data = np.c_[ze[:-2], ze[2:]]
    np.savetxt("../../data/lorenz_map_bis.dat", data)


    te, ze = te[te>50]-50, ze[te>50]
    te, ze = te[te<=30], ze[te<=30]

    data = np.c_[te, ze]
    np.savetxt("../../data/lorenz_events.dat", data)
    z = output["y"][-1]
    t, z = t[t>50]-50, z[t>50]
    t, z = t[t<=30], z[t<=30]
    data = np.c_[t, z]

    plt.plot(data[:, 0], data[:, 1])
    plt.show()
    np.savetxt("../../data/lorenz_z_time_series.dat", data)


if __name__ == "__main__":
    lorenz_map()
