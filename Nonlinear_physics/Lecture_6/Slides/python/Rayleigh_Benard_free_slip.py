import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from numpy.random import uniform, normal

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams.update(params)

w = 5.33

if __name__ == '__main__':

    def dispersion_relation(Ra, k, Pr=1, n=1):
        """
        Computation of the dispersion relation for the Rayleigh-Benard instability
        problem. Note that free-slip boundary conditions are assumed for the
        velocity perturbation.
        """

        p = np.zeros((2,))
        p[0] = (n**2 * np.pi**2 + k**2)*(Pr + 1.)
        p[1] = Pr*(k**2*Ra/(n**2*np.pi**2 + k**2) - (n**2*np.pi**2 + k**2)**2)

        s = np.roots(p)

        return s

    ra = np.linspace(200, 2200, 1000)
    k = np.linspace(0, 10, 100)

    ra, k = np.meshgrid(ra, k)
    s = np.zeros_like(ra)

    for i in xrange(s.size):
        s.ravel()[i] = dispersion_relation(ra.ravel()[i], k.ravel()[i])

    fig = plt.figure(figsize=(w/3, w/3))
    ax = fig.gca()

    ax.contour(k, ra, s, levels=[0.], colors='royalblue')

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$Ra$')

    ax.set_ylim(200, 2200)
    ax.locator_params(axis='y', nbins=5)

    plt.savefig('../imgs/rayleigh_benard_dispersion_relation.pdf', bbox_inches='tight', dpi=300)








    k = np.pi/np.sqrt(2)
    l = 2*np.pi/k
    x = np.linspace(0, 3, 40)
    y = np.linspace(0, 1, 20)
    x, y = np.meshgrid(x, y)

    u = np.pi*np.cos(np.pi*y)*np.sin(k*x)
    v = -k*np.sin(np.pi*y)*np.cos(k*x)
    theta = np.sin(np.pi*y) * np.cos(k*x)

    fig = plt.figure(figsize=(w, w/2))
    ax = fig.gca()

    ax.quiver(x, y, u, v, color='gray')
    ax.contour(x, y, theta, cmap=plt.cm.RdBu)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/rayleigh_benard_instability_mode.pdf', bbox_inches='tight', dpi=300)
    plt.show()
