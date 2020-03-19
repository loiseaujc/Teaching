import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

FIG_WIDTH = 5.33

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams.update(params)

def dynamics(t, x):

        # -->
        k, omega = 0.01, 1.0

        # -->
        dx = np.zeros_like(x)

        dx[0] = x[1]
        dx[1] = -2*k * x[1] - omega**2 * np.sin(x[0])

        return dx

def linearized_upward_dynamics(t, x):
        # -->
        k, omega = 0.01, 1.0

        # -->
        dx = np.zeros_like(x)

        dx[0] = x[1]
        dx[1] = -2*k * x[1] + omega**2 * x[0]

        return dx

def linearized_downward_dynamics(t, x):
        # -->
        k, omega = 0.075, 1.0

        # -->
        dx = np.zeros_like(x)

        dx[0] = x[1]
        dx[1] = -2*k * x[1] - omega**2 * x[0]

        return dx

if __name__ == "__main__":

        x = np.linspace(-1, 1, 16)
        y = np.linspace(-1, 1, 16)

        x, y = np.meshgrid(x, y)
        dx, dy = np.zeros_like(x), np.zeros_like(y)
        dx[:], dy[:] = linearized_downward_dynamics(0, [x[:], y[:]])

        # -->
        _, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH/3, FIG_WIDTH/3))
        ax.streamplot(
                x, y, dx, dy,
                color=np.sqrt(dx**2 + dy**2),
                cmap="inferno_r"
                )
        ax.set_aspect("equal")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        ax.locator_params(axis="both", nbins=3)

        plt.savefig("pendulum_downward_stab.png", bbox_inches="tight", dpi=300)




        dx[:], dy[:] = linearized_upward_dynamics(0, [x[:], y[:]])

        # -->
        x += np.pi
        _, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH/3, FIG_WIDTH/3))
        ax.streamplot(
                x, y, dx, dy,
                color=np.sqrt(dx**2 + dy**2),
                cmap="inferno_r"
                )
        ax.set_aspect("equal")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        ax.set_xlim(np.pi-1, np.pi+1)
        ax.set_xticks([np.pi-1, np.pi, np.pi+1])
        ax.set_xticklabels([r"$\pi-\epsilon$", r"$\pi$", r"$\pi + \epsilon$"])

        ax.set_ylim(-1, 1)

        ax.locator_params(axis="both", nbins=3)

        plt.savefig("pendulum_upward_stab.png", bbox_inches="tight", dpi=300)

        plt.show()
