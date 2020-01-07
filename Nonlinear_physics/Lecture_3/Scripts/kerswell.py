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

def dynamical_system(x, t=None):

    # --> Initialize variable.
    dx = np.zeros_like(x)

    # --> x-equation.
    dx[0] = -x[0] + 10.*x[1]

    # --> y-equation.
    dx[1] = x[1] * (10.*np.exp(-0.01*x[0]**2) - x[1]) * (x[1] - 1.)

    return dx

if __name__ == '__main__':

    # --> Mesh of the phase plane.
    x = np.linspace(-5, 20, 1000)
    y = np.linspace(-1, 3)

    x, y = np.meshgrid(x, y)

    # --> Compute xdot and ydot.
    xdot, ydot = np.zeros_like(x), np.zeros_like(x)

    xdot[:], ydot[:] = dynamical_system([x[:], y[:]])

    # --> Fixed points.
    xf = np.array([14.017, 0., 10.])
    yf = np.array([1.4017, 0., 1.])

    # --> Plot phase space.

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    magnitude = np.sqrt(xdot**2. + ydot**2.)
    ax.streamplot(x, y, xdot, ydot, color=magnitude, cmap=plt.cm.inferno)

    ax.plot(xf, yf, 'ro')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/kerswell_phase_plane.pdf', bbox_inches='tight', dpi=300)


    # --> Plot phase space.

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    magnitude = np.sqrt(xdot**2. + ydot**2.)
    ax.streamplot(x, y, xdot, ydot, color=magnitude, cmap=plt.cm.inferno, density=0.5)

    # --> Simulation.
    from scipy.integrate import odeint
    t = np.linspace(0, 10, 1000)
    x0 = np.asarray([xf[-1], yf[-1]+1e-6])
    sol = odeint(dynamical_system, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], color='orange', lw=2)
    x0 = np.asarray([xf[-1], yf[-1]-1e-6])
    sol = odeint(dynamical_system, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], color='orange', lw=2)

    ax.axhline(1, color='royalblue', lw=2)
    ax.plot(xf, yf, 'ro')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/kerswell_phase_plane_bis.pdf', bbox_inches='tight', dpi=300)



    # --> Different initial condition.


    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    t = np.linspace(0, 10, 1000)
    x0 = np.asarray([xf[-1], yf[-1]+1e-6])
    sol = odeint(dynamical_system, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], color='gray', lw=2)
    x0 = np.asarray([xf[-1], yf[-1]-1e-6])
    sol = odeint(dynamical_system, x0, t)
    ax.plot(sol[:, 0], sol[:, 1], color='gray', lw=2)

    ax.axhline(1, color='gray', lw=2)

    ax.plot(xf, yf, 'ro')

    ax.set_xlim(-5, 20)
    ax.set_ylim(-1, 3)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    Y0 = np.array([0.25, 0.5, 0.75, 0.990, 1.00001])
    for i, y0 in enumerate(Y0):
        x0 = np.array([0., y0])
        sol = odeint(dynamical_system, x0, t)
        ax.plot(sol[:, 0], sol[:, 1], color='royalblue')
        ax.plot(x0[0], x0[1], '.', color='green')
        plt.savefig('../imgs/kerswell_phase_plane_ter_%i.pdf' %i, bbox_inches='tight', dpi=300)


    plt.show()
