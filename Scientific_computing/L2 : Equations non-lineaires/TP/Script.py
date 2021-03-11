import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

sec = lambda θ: 1/np.cos(θ)

def trajectoire(x, θ, α=1e-1):
    y = ( np.tan(θ) + sec(θ)/α ) * x + 1/α**2 * np.log(1 - α*sec(θ)*x)
    return y

@np.vectorize
def impact(θ, α=1e-1):
    F = lambda x : ( α**2*np.tan(θ) + α*sec(θ) ) * x + np.log(1 - α*sec(θ)*x)
    return root_scalar(F, method="bisect", bracket=[0.01, 1.01]).root

if __name__ == "__main__":

    θs = np.linspace(np.pi/32, 14*np.pi/32, 14)

    α = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, θ in enumerate(θs[::-1]):

        x = np.linspace(0, np.cos(θ)/α, 4096)[:-1]
        y = trajectoire(x, θ, α=α)
        ax.plot(
            x, y,
            color=plt.cm.viridis(i/len(θs)),
            )

    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Distance x")
    ax.set_ylabel("Altitude y")

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("imgs/trajectories.png", bbox_inches="tight", dpi=300)

    θs = np.linspace(np.pi/32, 14*np.pi/32, 128)

    D = impact(θs, α=α)

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    ax.plot(θs*180/np.pi, D, color="black")

    ax.set(xlim=(0, 90), xlabel="Angle de hausse θ (en degré)")
    ax.set(ylim=(0, 1.025), ylabel="Distance parcourue")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("imgs/distance_parcourue.png", bbox_inches="tight", dpi=300)

    plt.show()
