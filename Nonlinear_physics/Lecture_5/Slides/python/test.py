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

from pynamical import cobweb_plot, simulate, phase_diagram, bifurcation_plot

if __name__ == '__main__':

    # mu = np.linspace(1, 3.9, 10)
    #
    # for i, r in enumerate(mu):
    #
    #     fig, ax = cobweb_plot(r = r,
    #                           figsize = (fig_width/3, fig_width/3),
    #                           dpi = 1200,
    #                           bbox_inches = 'tight',
    #                           show=False
    #                           )
    #
    #     ax.set_title('')
    #     ax.set_xlabel(r'$x$')
    #     ax.set_ylabel(r'$\mathcal{G}(x)$')
    #     ax.set_aspect('equal')
    #
    #     plt.savefig('../imgs/logistic_map_cobweb_plot_%i.pdf' %i, bbox_inches='tight', dpi=1200)
    #     plt.close()
    #
    #
    # for i, r in enumerate(mu):
    #
    #     pops = simulate(num_gens=3000, rate_min=r, num_rates=1, num_discard=1000)
    #
    #     fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    #     ax = fig.gca()
    #
    #     ax.plot(pops[:-1], pops[1:], '.', ms=1, color='royalblue')
    #
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #
    #     ax.set_aspect('equal')
    #
    #     ax.set_xlabel(r'$x_k$')
    #     ax.set_ylabel(r'$x_{k+1}$')
    #
    #     plt.savefig('../imgs/logistic_map_phase_plane_%i.pdf' %i, bbox_inches='tight', dpi=1200)
    #     plt.close()
    #
    #
    #
    #
    #
    # x = simulate(num_gens=30000, rate_min=3.99, num_rates=1, num_discard=29000)
    # x = x.as_matrix()[29500:]
    # from numpy.random import uniform
    #
    # y = uniform(low=x.min(), high=x.max(), size=x.shape)
    #
    # fig = plt.figure(figsize=(fig_width, fig_width/3))
    # ax = fig.gca()
    #
    # ax.plot(x, color='royalblue', label=r'Chaos')
    # ax.plot(y, color='orange', label=r'Random')
    #
    # ax.set_xlim(40, 90)
    # ax.set_ylim(0, 1)
    #
    # ax.set_xlabel(r'Index $k$')
    # ax.set_ylabel(r'$x_k$')
    #
    # plt.savefig('../imgs/chaos_vs_randomness.pdf', bbox_inches='tight', dpi=300)
    #
    # fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    # ax = fig.gca()
    #
    # ax.plot(x[:-1], x[1:], '.', ms=1, color='royalblue')
    # ax.plot(y[:-1], y[1:], '.', ms=1, color='orange')
    #
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    #
    # ax.set_aspect('equal')
    #
    # ax.set_xlabel(r'$x_k$')
    # ax.set_ylabel(r'$x_{k+1}$')
    #
    # plt.savefig('../imgs/chaos_vs_randomness_phase_plane.pdf', bbox_inches='tight', dpi=300)







    pops = simulate(num_gens=100, rate_min=0, rate_max=4, num_rates=1000, num_discard=100, jit=True)
    pops = pops.as_matrix()
    mu = np.linspace(0, 4, 1000)

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    for i in xrange(mu.size):
        ax.plot(mu[i]*np.ones_like(pops[:, i]), pops[:, i], ',', color='royalblue')


    # --> x-axis.
    ax.set_xlim(0, 4)
    ax.set_xlabel(r'$\mu$')

    # --> y-axis.
    ax.set_ylim(0, 1)
    ax.set_ylabel(r'$x^*$')

    plt.savefig('../imgs/logistic_map_bifurcation_overview.pdf', bbox_inches='tight', dpi=300)




    pops = simulate(num_gens=100, rate_min=2.8, rate_max=4, num_rates=1000, num_discard=200, jit=True)
    pops = pops.as_matrix()
    mu = np.linspace(2.8, 4, 1000)

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    for i in xrange(mu.size):
        ax.plot(mu[i]*np.ones_like(pops[:, i]), pops[:, i], ',', color='royalblue')


    # --> x-axis.
    ax.set_xlim(mu.min(), mu.max())
    ax.set_xlabel(r'$\mu$')

    # --> y-axis.
    ax.set_ylim(0, 1)
    ax.set_ylabel(r'$x^*$')

    plt.savefig('../imgs/logistic_map_bifurcation_zoom_1.pdf', bbox_inches='tight', dpi=300)







    pops = simulate(num_gens=100, rate_min=3.7, rate_max=3.9, num_rates=1000, num_discard=200, jit=True)
    pops = pops.as_matrix()
    mu = np.linspace(3.7, 3.9, 1000)

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    for i in xrange(mu.size):
        ax.plot(mu[i]*np.ones_like(pops[:, i]), pops[:, i], ',', color='royalblue')


    # --> x-axis.
    ax.set_xlim(mu.min(), mu.max())
    ax.set_xlabel(r'$\mu$')

    # --> y-axis.
    ax.set_ylim(0, 1)
    ax.set_ylabel(r'$x^*$')

    plt.savefig('../imgs/logistic_map_bifurcation_zoom_2.pdf', bbox_inches='tight', dpi=300)




    pops = simulate(num_gens=200, rate_min=3.84, rate_max=3.856, num_rates=1000, num_discard=200, jit=True)
    pops = pops.as_matrix()
    mu = np.linspace(3.84, 3.856, 1000)

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    for i in xrange(mu.size):
        ax.plot(mu[i]*np.ones_like(pops[:, i]), pops[:, i], ',', color='royalblue')


    # --> x-axis.
    ax.set_xlim(mu.min(), mu.max())
    ax.set_xlabel(r'$\mu$')

    # --> y-axis.
    ax.set_ylim(0.445, 0.552)
    ax.set_ylabel(r'$x^*$')

    plt.savefig('../imgs/logistic_map_bifurcation_zoom_3.pdf', bbox_inches='tight', dpi=300)

    plt.show()
