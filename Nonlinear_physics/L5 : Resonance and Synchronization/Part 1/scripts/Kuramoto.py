import numpy as np

from scipy.integrate import solve_ivp
from scipy.stats import cauchy, norm

from matplotlib import pyplot as plt

from numba import jit

def kuramoto(t, θ, p):

    # --> Extract parameters.
    ω, k = p # Natural frequency, coupling strength.

    # --> Number of oscillators.
    n = len(θ)

    dθ = np.zeros_like(θ)

    # --> Kuramoto model.
    for i in range(n):
        dθ[i] = ω[i]
        for j in range(n):
            dθ[i] += (k/n) * np.sin(θ[j] - θ[i])

    return dθ

if __name__ == "__main__":

    # --> Parameters of the system.
    n = 200 # Number of oscillators.
    ω = cauchy.rvs(loc=0, scale=0.5, size=n)
    k = 2.0 # Coupling strength.

    p = [ω, k]

    # --> Parameters of the simulation.
    θ = 2*np.pi * np.random.randn(n) # Initial condition.
    tspan = (0.0, 20.0)
    t_eval = np.linspace(*tspan, 1024*4)

    # --> Run the simulation.
    output = solve_ivp(
        lambda t, θ : kuramoto(t, θ, p),
        tspan,
        θ,
        t_eval=t_eval
        )

    # --> Extract the solution.
    θ = output["y"].T
    R = np.array([np.mean( np.exp(1j * θ[i])) for i in range(len(t_eval))])
    r, Ψ = np.abs(R), np.angle(R)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(t_eval, r)
    axes[1].plot(t_eval, Ψ)

    plt.show()
