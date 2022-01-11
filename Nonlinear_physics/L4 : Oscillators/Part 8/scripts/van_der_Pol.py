import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

def dynsys(t, u, ϵ):

    # --> Unpack variables.
    x, z = u

    # --> Governing equations.
    dx = (x - x**3/3 - z) / ϵ
    dz = ϵ * x

    return dx, dz





def main():

    # -->
    tspan = (0, 100)
    x0 = [0.25, 0]
    t = np.linspace(*tspan, 1024)
    ϵ = 10

    # --> Simulate the system.
    output = solve_ivp(
        lambda t, x : dynsys(t, x, ϵ),
        tspan,
        x0,
        t_eval=t
        )

    x = output["y"]
    x[0] /= 0.5 * x[0].max()
    x[1] /= x[1].max()
    t = t/t.max() * 6.5

    plt.plot(x[0], x[1])
    plt.show()

    np.savetxt("../traj_1.txt", np.asarray([t, x[0]/x[0].max()]).T)
    np.savetxt("../phase_portrait_1.txt", x.T)





    tspan = (0, 50)
    t = np.linspace(*tspan, 1024)
    ϵ = 1

    # --> Simulate the system.
    output = solve_ivp(
        lambda t, x : dynsys(t, x, ϵ),
        tspan,
        x0,
        t_eval=t
        )

    x = output["y"]
    x[0] /= 0.5 * x[0].max()
    x[1] /= x[1].max()
    t = t/t.max() * 6.5

    plt.plot(x[0], x[1])
    plt.show()

    np.savetxt("../traj_2.txt", np.asarray([t, x[0]/x[0].max()]).T)
    np.savetxt("../phase_portrait_2.txt", x.T)


    tspan = (0, 100)
    t = np.linspace(*tspan, 1024)
    ϵ = 0.1

    # --> Simulate the system.
    output = solve_ivp(
        lambda t, x : dynsys(t, x, ϵ),
        tspan,
        x0,
        t_eval=t
        )

    x = output["y"]
    print(output["y"].max(axis=1))
    x[0] /= 0.5 * x[0].max()
    x[1] /= x[1].max()
    t = t/t.max() * 6.5

    plt.plot(x[0], x[1])
    plt.show()

    np.savetxt("../traj_3.txt", np.asarray([t, x[0]/x[0].max()]).T)
    np.savetxt("../phase_portrait_3.txt", x.T)

    dx = np.array([dynsys(0, x, ϵ) for x in output["y"].T])
    print(dx.shape)

    plt.plot(dx[:, 0])
    plt.show()


if __name__ == "__main__":

    main()
