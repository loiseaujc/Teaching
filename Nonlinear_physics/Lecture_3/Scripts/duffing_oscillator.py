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

def duffing_oscillator(x, t=None):

    # --> Initialize variables.
    dx = np.zeros_like(x)

    # --> x-equation.
    dx[0] = x[1]

    # --> y-equation.
    dx[1] = -0.5*x[1] + x[0] - x[0]**3.

    return dx

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

def duffing_jacobian(x):

    x = np.asarray(x)

    # --> Initialize variable.
    A = np.zeros((x.shape[0], x.shape[0]))

    # --> x-equation.
    A[0, 0] = 0.
    A[0, 1] = 1.

    # --> y-equation.
    A[1, 0] = 1. - 3*x[0]**2
    A[1, 1] = -0.5

    return A

if __name__ == '__main__':

    # --> Definition of the grid for the phase plane.
    x = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, x)

    # --> Compute the time derivative vectors.
    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = duffing_oscillator([x[:], y[:]])

    # --> Determine the fixed points.
    xf = np.array([0., 1., -1.])
    yf = np.zeros_like(xf)

    # --> Plot the figure.
    streamlines_and_isoclines(x, y, xdot, ydot, xf, yf, savename='duffing_phase_plane')



    # --> Numerical integration.
    from scipy.integrate import odeint
    from numpy.random import uniform
    t = np.linspace(0, 100, 1000.)

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()


    for i in xrange(8):
        x0 = uniform(low=-2, high=2, size=(2,))
        x = odeint(duffing_oscillator, x0, t)

        if x[-1, 0] > 0:
            ax.plot(x[:, 0], x[:, 1], color='royalblue')
        else:
            ax.plot(x[:, 0], x[:, 1], color='orange')

    ax.plot(xf, yf, 'r.')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/duffing_oscillator_basin.pdf', bbox_inches='tight', dpi=300)

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()


    for i in xrange(100):
        x0 = uniform(low=-2, high=2, size=(2,))
        x = odeint(duffing_oscillator, x0, t)

        if x[-1, 0] > 0:
            ax.plot(x[:, 0], x[:, 1], color='royalblue')
        else:
            ax.plot(x[:, 0], x[:, 1], color='orange')

    ax.plot(xf, yf, 'r.')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/duffing_oscillator_basin_bis.pdf', bbox_inches='tight', dpi=300)

    # --> Stable and unstable manifolds.
    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    t = np.linspace(0, 35, 1000)

    from scipy.linalg import eig
    A = duffing_jacobian([xf[0], yf[0]])
    _, eigenvectors = eig(A)

    x0 = 1e-8*eigenvectors[:, 1]
    x = odeint(duffing_oscillator, x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue')
    x = odeint(duffing_oscillator, -x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue')

    x0 = 1e-8*eigenvectors[:, 0]
    x = odeint(duffing_oscillator, x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange')
    x = odeint(duffing_oscillator, -x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange')

    ax.plot(xf, yf, 'r.')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/duffing_oscillator_saddle_manifold.pdf', bbox_inches='tight', dpi=300)

    # --> Stable and unstable manifolds.
    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    t = np.linspace(0, 35, 1000)

    from scipy.linalg import eig
    A = duffing_jacobian([xf[0], yf[0]])
    _, eigenvectors = eig(A)

    x0 = 1e-8*eigenvectors[:, 1]
    x = odeint(duffing_oscillator, x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue', alpha=0.5)
    x = odeint(duffing_oscillator, -x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue', alpha=0.5)


    x0 = 1e-8*eigenvectors[:, 0]
    x = odeint(duffing_oscillator, x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange', alpha=0.5)
    x = odeint(duffing_oscillator, -x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange', alpha=0.5)


    ax.plot(xf, yf, 'r.')

    ax.arrow(0, 0, 0.5*eigenvectors[0, 1], 0.5*eigenvectors[1, 1], head_width=0.05, head_length=-0.1, fc='k', ec='k')
    ax.arrow(0, 0, -0.5*eigenvectors[0, 1], -0.5*eigenvectors[1, 1], head_width=0.05, head_length=-0.1, fc='k', ec='k')

    ax.arrow(0, 0, 0.25*eigenvectors[0, 0], 0.25*eigenvectors[1, 0], head_width=0.05, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, -0.25*eigenvectors[0, 0], -0.25*eigenvectors[1, 0], head_width=0.05, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/duffing_oscillator_saddle_manifold_bis.pdf', bbox_inches='tight', dpi=300)


    # --> Stable and unstable manifolds.
    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    t = np.linspace(0, 35, 1000)

    from scipy.linalg import eig
    A = duffing_jacobian([xf[0], yf[0]])
    _, eigenvectors = eig(A)

    x0 = 1e-8*eigenvectors[:, 1]
    x = odeint(duffing_oscillator, x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue')
    x = odeint(duffing_oscillator, -x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue')


    x0 = 1e-8*eigenvectors[:, 0]
    x = odeint(duffing_oscillator, x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange')
    x = odeint(duffing_oscillator, -x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    a = -(1./4.) * (1. + np.sqrt(17.))
    c = -1./(4.*a + 0.5)

    xx = np.linspace(-1.5, 1.5)
    yy = a*xx + c*xx**3

    plt.plot(xx, yy, '--', color='gray')

    a = -(1./4.) * (1. - np.sqrt(17.))
    c = -1./(4.*a + 0.5)

    yy = a*xx + c*xx**3

    plt.plot(xx, yy, '--', color='gray')

    ax.plot(xf, yf, 'r.')

    plt.savefig('../imgs/duffing_oscillator_saddle_manifold_ter.pdf', bbox_inches='tight', dpi=300)



    # --> Plot the total energy.

    x = np.linspace(-2, 2)
    x, y = np.meshgrid(x, x)

    H = -0.5*x**2 + 0.25*x**4 + 0.5*y**2

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    levels = np.linspace(-0.2, 2, 10)
    ax.contour(x, y, H, levels=levels, vmax=2.)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/duffing_oscillator_potential.pdf', bbox_inches='tight', dpi=300)

    plt.show()
