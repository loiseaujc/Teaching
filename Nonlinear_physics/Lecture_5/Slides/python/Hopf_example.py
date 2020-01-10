import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D

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
    sigma = 0.1
    def dynamical_system(x, t=None):
        # --> Initialize variables.
        dx = np.zeros_like(x)

        # --> x-equation.
        dx[0] = sigma*x[0] - x[1] - x[0]*x[2]
        # --> y-equation.
        dx[1] = x[0] + sigma*x[1] - x[1]*x[2]
        # --> z-equation.
        dx[2] = -x[2] + x[0]**2 + x[1]**2

        return dx

    # --> Simulate model.
    t = np.linspace(0, 100, 1000)
    x0 = np.array([1e-3, 0, 0])

    from scipy.integrate import odeint
    sol = odeint(dynamical_system, x0, t)

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca(projection='3d')

    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], color='royalblue')

    ax.set_xticklabels([])
    ax.set_xlabel(r'$x$', labelpad=-10)
    ax.set_yticklabels([])
    ax.set_ylabel(r'$y$', labelpad=-10)
    ax.set_zticklabels([])
    ax.set_zlabel(r'$z$', labelpad=-10)

    plt.savefig('../imgs/generalized_mean_field_model.pdf', bbox_inches='tight', dpi=300)


    ######

    x = np.linspace(-0.5, 0.5)
    z = np.linspace(0, 0.25)

    x, z = np.meshgrid(x, z)

    dx, dz = np.zeros_like(x), np.zeros_like(z)

    dx[:], _, dz[:] = dynamical_system([x[:], np.zeros_like(x)[:], z[:]])

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    magnitude = np.sqrt(dx**2 + dz**2)
    ax.streamplot(x, z, dx, dz, color=magnitude, cmap=plt.cm.inferno)
    x = np.linspace(-0.5, 0.5)
    z = x**2
    z /= 2*sigma + 1.
    ax.plot(x, z, color='red', lw=2)

    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylim(0, 0.2)
    ax.set_ylabel(r'$z$')

    plt.savefig('../imgs/generalized_mean_field_model_manifold.pdf', bbox_inches='tight', dpi=300)


    #####

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    ax.plot(sol[:, 0], sol[:, 1], color='royalblue')
    ax.plot(0, 0, marker='o', color='orange', fillstyle='none', markeredgewidth=2, ms=8)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_aspect('equal')

    plt.savefig('../imgs/generalized_mean_field_model_2D.pdf', bbox_inches='tight', dpi=300)
    plt.show()
