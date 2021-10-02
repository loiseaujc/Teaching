import numpy as np
import math

from scipy.integrate import solve_ivp
from scipy.fft import fft

import matplotlib.pyplot as plt

def dynsys(t, u, p):

    # --> Unpack variables.
    x, dx = u

    # --> Unpack parameters.
    δ, β, γ, ω = p

    # --> Equation of motion.
    ddx = -δ*dx - x - β*x**3 + γ * math.cos(ω*t)

    return dx, ddx

def main(δ=0.1, β=0.04, γ=1.0, n_periods=200, nfreqs=256):

    # --> Frequency range.
    Ω = np.linspace(0.1, 3, nfreqs)

    # --> Initial condition for the very first simulation.
    x0 = np.zeros(2)

    # --> Arrays to store the amplitude of the maximum Fourier coefficient.
    up, down = np.zeros_like(Ω), np.zeros_like(Ω)

    # --> Upward sweep.
    for i, ω in enumerate(Ω):

        # --> Parameters.
        p = [δ, β, γ, ω]

        # --> Sampling period.
        ΔT = 2*np.pi / ω
        t = np.arange(n_periods) * ΔT
        tspan = (t.min(), t.max())

        # --> Run simulation.
        output = solve_ivp(
            lambda t, x : dynsys(t, x, p),
            tspan,
            x0,
            t_eval=t
        )

        # --> Extract solution and postprocess.
        x = output["y"][:, n_periods//4:]
        x0 = output["y"][:, -1]
        x̂ = fft(x, axis=-1)
        z = np.array([np.linalg.norm(y) for y in x.T])
        up[i] = z.max()

    # --> Initial condition for the very first simulation.
    x0 = np.zeros(2)

    # --> Upward sweep.
    for i, ω in enumerate(Ω[::-1]):

        # --> Parameters.
        p = [δ, β, γ, ω]

        # --> Sampling period.
        ΔT = 2*np.pi / ω
        t = np.arange(n_periods) * ΔT
        tspan = (t.min(), t.max())

        # --> Run simulation.
        output = solve_ivp(
            lambda t, x : dynsys(t, x, p),
            tspan,
            x0,
            t_eval=t
        )

        # --> Extract solution and postprocess.
        x = output["y"][:, n_periods//4:]
        x0 = output["y"][:, -1]
        x̂ = fft(x, axis=-1)
        z = np.array([np.linalg.norm(y) for y in x.T])
        down[i] = z.max()

    return Ω, up, down[::-1]


if __name__ == "__main__":

    Ω, up, down = main()

    fig, ax = plt.subplots(1, 1)
    ax.plot(Ω, up, ".")
    ax.plot(Ω, down, "x")

    plt.show()
