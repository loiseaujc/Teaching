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

def streamlines_and_isoclines(x, y, u, v, xf=None, yf=None, savename=None):

    fig, ax = plt.subplots(1, 2, figsize=(fig_width, fig_width/3))

    streamlines(ax[0], x, y, u, v)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$', rotation=0)

    isoclines(ax[1], x, y, u, v)

    if (xf is not None) and (yf is not None):
        ax[0].plot(xf, yf, 'ro')
        ax[1].plot(xf, yf, 'ro')

    if savename is not None:
        plt.savefig('../imgs/'+savename+'.pdf', bbox_inches='tight', dpi=300)

    return

def streamlines(ax, x, y, u, v, density=1):

    #-->
    magnitude = np.sqrt(u**2 + v**2)

    ax.streamplot(x, y, u, v, color=magnitude, cmap=plt.cm.inferno, density=density)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_aspect('equal')
    ax.set_title(r'Streamlines')

    return

def isoclines(ax, x, y, u, v):

    ax.quiver(x[::4, ::4], y[::4, ::4], u[::4, ::4], v[::4, ::4], color='k')
    c1 = ax.contour(x, y, u, levels=[0.], colors=colors[1], label=r'$\dot{x}=0$')
    c2 = ax.contour(x, y, v, levels=[0.], colors=colors[2], label=r'$\dot({y}=0$')

    fmt = {}
    strs = [r'$\dot{x}=0$']
    for l, s in zip(c1.levels, strs):
        fmt[l] = s
    ax.clabel(c1, c1.levels, inline=True, fmt=fmt, fontsize=8)

    fmt = {}
    strs = [r'$\dot{y}=0$']
    for l, s in zip(c2.levels, strs):
        fmt[l] = s
    ax.clabel(c2, c2.levels, inline=True, fmt=fmt, fontsize=8)

    ax.set_xlabel(r'$x$')

    ax.set_yticklabels([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_aspect('equal')
    ax.set_title(r'Isoclines')

    return

if __name__ == '__main__':

    def dynamical_system(x, t=None):

        dx = np.zeros_like(x)

        dx[0] = -x[0]

        dx[1] = 2.*x[1] - 5.*x[0]**3.

        return dx

    # --> Definition of the grid for the phase plane.
    x = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, x)

    # --> Compute the time derivative vectors.
    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = dynamical_system([x[:], y[:]])

    # --> Determine the fixed points.
    xf = np.array([0.])
    yf = np.zeros_like(xf)

    # --> Plot the figure.
    streamlines_and_isoclines(x, y, xdot, ydot, xf, yf, savename='exercise_phase_plane')


    def dynamical_system(x, t=None):

        dx = np.zeros_like(x)

        dx[0] = -x[0] + x[1]**2.

        dx[1] = x[1] - x[0]**2.

        return dx

    # --> Definition of the grid for the phase plane.
    x = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, x)

    # --> Compute the time derivative vectors.
    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = dynamical_system([x[:], y[:]])

    # --> Determine the fixed points.
    xf = np.array([0., 1.])
    yf = np.array([0., 1.])

    # --> Plot the figure.
    streamlines_and_isoclines(x, y, xdot, ydot, xf, yf, savename='exercise_phase_plane_bis')

    plt.show()
