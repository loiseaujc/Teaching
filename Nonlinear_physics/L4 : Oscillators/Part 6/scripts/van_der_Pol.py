import numpy as np

from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

def van_der_Pol(t, u, ϵ):

    # --> Unpack variables.
    x, dx, r = u

    # --> van der Pol oscillator.
    ddx = -x - ϵ*(x**2 - 1)*dx

    # --> Amplitude equation.
    dr = ϵ/8 * r * (4 - r**2)

    return dx, ddx, dr

def main():

    # --> Parameters of the simulation.
    tspan = (0, 100)
    x0 = [0.25, 0, 0.25]
    t = np.linspace(*tspan, 1024)
    ϵ = 0.1

    # --> Simulate the system.
    output = solve_ivp(
        lambda t, x : van_der_Pol(t, x, ϵ),
        tspan,
        x0,
        t_eval=t
        )

    x = output["y"][0] * (1.5/2)
    r = output["y"][-1] * (1.5/2)
    t = t / 10

    np.savetxt("../traj_x.txt", np.asarray([t, x]).T)
    np.savetxt("../traj_r.txt", np.asarray([t, r]).T)

    return

if __name__ == "__main__":
    main()
