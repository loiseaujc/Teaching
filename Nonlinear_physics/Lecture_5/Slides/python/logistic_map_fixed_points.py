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


    mu = np.array([1., 2.5, 4.])
    # --> x-axis.
    x = np.linspace(0, 1)

    # --> Define the logistic map.
    f = lambda x, r: r*x*(1-x)

    # --> Plot the figure.
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_width/3))

    for i, r in enumerate(mu):
        ax = axes[i]

        ax.plot(x, f(x, r), color=colors[0])
        ax.plot(x, x, color=colors[0])
        ax.plot((r-1.)/r, (r-1.)/r, 'o', color=colors[1])

        # --> Decorators.
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        ax.set_xlabel(r'$x_k$')
        if i == 0:
            ax.set_ylabel(r'$\mathcal{G}(x)$')
        else:
            ax.set_yticklabels([])

        ax.set_title(r'$\mu = %.2f$' %r, y=-0.5)

    plt.savefig('../imgs/logistic_map_fixed_points.pdf', bbox_inches='tight', dpi=300)
    plt.show()
