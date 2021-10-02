import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

def stuart_landau_oscillator(t, η, p):

    # -->
    λ, β = p

    # -->
    dη = λ*η - β*np.abs(η)**2 * η

    return dη

def dynsys(t, η, p):

    n = len(η)

    λ, β, k = p

    # --> Uncoupled dynamics.
    dη = np.array([stuart_landau_oscillator(t, η[i], [λ[i], β[i]]) for i in range(n)])

    # --> Coupling term.
    for i in range(n):
        for j in range(n):
            if i != j:
                dη[i] += (k/n)*η[j]

    return dη

if __name__ == "__main__":

    n = 64
    σ, ω = 0.1 + 0.01*np.random.randn(n), 1.0 + 0.1*np.random.randn(n)
    λ, β = σ + 1j*ω, np.ones(n)
    k = 0.165
    p = [λ, β, k]

    η = np.random.randn(n) + 1j*np.random.randn(n)
    η /= np.abs(η)

    tspan = (0.0, 1000.0)
    t = np.linspace(*tspan, 4096)

    output = solve_ivp(
        lambda t, η : dynsys(t, η, p),
        tspan,
        η,
        t_eval=t
        )

    η = output["y"].T
    θ = np.angle(η)
    R = np.array([np.mean( np.exp(1j * θ[i])) for i in range(len(t))])
    r, Ψ = np.abs(R), np.angle(R)

    print(r.mean(), r.std())

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(t, r)
    axes[1].plot(t, Ψ)

    plt.show()
