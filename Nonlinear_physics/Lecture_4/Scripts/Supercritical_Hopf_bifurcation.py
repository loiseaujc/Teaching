import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams.update(params)
colors = [ 'dimgrey', 'royalblue', 'orange', 'seagreen', 'y' ]

fig_width = 5.33


if __name__ == '__main__':

    # --> Define the system.
    def dynamical_system(x, t=None, mu=1, omega=1):
        # --> Initialize variable.
        dx = np.zeros_like(x)

        # --> x-equation.
        dx[0] = mu*x[0] - omega*x[1] - (x[0]**2 + x[1]**2)*x[0]

        # --> y-equation.
        dx[1] = omega*x[0] + mu*x[1] - (x[0]**2 + x[1]**2)*x[1]

        return dx

    # --> Mesh of the phase plane.
    x = np.linspace(-1, 1)
    x, y = np.meshgrid(x, x)

    # --> Control parameter.
    mu = np.array([-0.5, 0, 0.5])

    # --> Plot the phase planes.
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_width/3))

    for i, ax in enumerate(axes):
        # --> Compute xdot and ydot.
        xdot, ydot = np.zeros_like(x), np.zeros_like(y)
        xdot[:], ydot[:] = dynamical_system([x[:], y[:]], mu=mu[i])

        # --> Plot streamlines.
        magnitude = np.sqrt(xdot**2 + ydot**2)
        ax.streamplot(x, y, xdot, ydot, color=magnitude, cmap=plt.cm.inferno, zorder=0)

        # --> Add fixed point.
        if mu[i] < 0:
            ax.plot(0, 0, marker='o', color='red', zorder=1)
        elif mu[i] > 0:
            ax.plot(0, 0, marker='o', color='red', fillstyle='none', zorder=1)
            theta = np.linspace(0, 2*np.pi)
            amplitude = np.sqrt(mu[i])
            ax.plot(amplitude*np.cos(theta), amplitude*np.sin(theta), color='r', lw=2, zorder=2)
        else:
            ax.plot(0, 0, marker='o', color='red', zorder=1)

        # --> Decorators for x axis.
        ax.set_xlabel(r'$x$')

        # --> Decorators for y axis.
        if i == 0:
            ax.set_ylabel(r'$y$')
        else:
            ax.set_yticklabels([])

        ax.set_aspect('equal')

        # --> Set the title.
        if mu[i] < 0:
            ax.set_title(r'$\mu < 0$', y=-.5)
        elif mu[i] == 0:
            ax.set_title(r'$\mu = 0$', y=-.5)
        else:
            ax.set_title(r'$\mu > 0$', y=-.5)

    plt.savefig('../imgs/supercritical_hopf_bifurcation_phase_plane.pdf', bbox_inches='tight', dpi=300)

    plt.show()
