import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import solve_ivp

def sir(t, u, R0):

    # --> Unpack variables.
    s, i, r = u

    # --> SIR model.
    ds = -R0 * s * i
    di = (R0*s - 1) * i
    dr = i
    return ds, di, dr

def plot(t, u):

    fig, ax = plt.subplots(1, 1, figsize=(8, 2), facecolor="black")

    ax.plot(t, u[0], label="S", color="dodgerblue", zorder=1)
    ax.plot(t, u[1], label="I", color="red", zorder=2)
    ax.plot(t, u[2], label="R", color="lightgray", zorder=0)

    ax.set_facecolor("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.set(xlim=(0, 10), ylim=(0, 100))
    ax.set(xlabel="Time", ylabel="Percent (%)")
    ax.tick_params(axis="both", colors="white")

    ax.legend(loc="upper right", facecolor="none", labelcolor="white")

    return fig, ax


def main():

    # --> Initial conditions.
    u = np.array([0.9999, 0.0001, 0.0])

    # --> Parameters.
    τ = 10
    R0 = 5

    # --> Simulate the system.
    output = solve_ivp(
        lambda t, u : sir(t, u, R0),
        (0, τ),
        u,
        dense_output=True
    )

    t = np.linspace(0, τ, 1024)
    fig, ax = plot(t, 100*output["sol"](t))
    plt.savefig("../imgs/sir_final.png", bbox_inches="tight", dpi=1200)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2), facecolor="black")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)

    sline, = ax.plot([], [], color="dodgerblue", label="S", zorder=1)
    iline, = ax.plot([], [], color="red", label="I", zorder=2)
    rline, = ax.plot([], [], color="lightgray", label="R", zorder=0)

    ax.set_facecolor("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.set(xlim=(0, 10), ylim=(0, 100))
    ax.set_xlabel("Time")
    ax.set_ylabel("Percent (%)")
    ax.tick_params(axis="both", colors="white")

    ax.legend(loc="upper right", facecolor="none", labelcolor="white")

    plt.tight_layout()

    def animate(k):
        s, i, r = 100*output["sol"](t[:k+1])
        sline.set_data(t[:k+1], s)
        iline.set_data(t[:k+1], i)
        rline.set_data(t[:k+1], r)

        return sline, iline, rline

    ani = animation.FuncAnimation(
        fig, animate, len(t), interval=1
    )

    ani.save("../imgs/sir_model.mp4", fps=30, extra_args=["-vcodec", "libx264"])

    return ani

if __name__ == "__main__":
    ani = main()
