import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

if __name__ == "__main__":

    σ = 0.128

    def dynsys(t, u):
        return σ*u - u**3

    tspan = (0.0, 150.0)
    t = np.linspace(*tspan, 2048)
    u0 = np.array([1e-4])

    r = solve_ivp(
        dynsys,
        tspan,
        u0,
        t_eval=t
    )["y"][0]

    r /= np.sqrt(σ)
    φ = t

    data = np.c_[r*np.cos(φ), r*np.sin(φ), r**2]
    np.savetxt("../trajectory.txt", data)
