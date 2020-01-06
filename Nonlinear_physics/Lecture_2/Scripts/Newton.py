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

    # --> Define the function and its derivative.
    f = lambda x : x**3. - 2.*x - 5.
    fp = lambda x : 3.*x**2. - 2.

    # --> Define the computational mesh.
    x = np.linspace(1, 4)

    # --> Run Newton method.
    x_ = [3.9]
    for i in xrange(3):
        x_.append(x_[-1] - f(x_[-1])/fp(x_[-1]))

    x_ = np.asarray(x_)

    #################################
    #####   Plot the figure     #####
    #################################

    # --> Setup the figure.
    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    # --> Plot the function itself.
    ax.plot(x, f(x), color=colors[0])

    # --> Plot the Newton iterates.
    ax.plot(x_, f(x_), 'o', color=colors[1])

    # --> Plot the vertical lines.
    lines = list()
    for i in xrange(x_.size):
        lines.append([(x_[i], 0.), (x_[i], f(x_[i]))])

    lc = mc.LineCollection(lines, colors='k', linestyles='--')
    ax.add_collection(lc)

    # --> Plot the derivative lines.
    lines = list()
    for i in xrange(x_.size-1):
        lines.append([(x_[i], f(x_[i])), (x_[i+1], 0)])

    lc = mc.LineCollection(lines, colors='k', linewidths=1)
    ax.add_collection(lc)

    # --> Add the xticks and xticklabels.
    labels = list()
    for i in xrange(x_.size):
        labels.append(r'$x_%i$' %i)

    ax.set_xticks(x_)
    ax.set_xticklabels(labels)

    ax.set_xlabel(r'$x$')
    ax.xaxis.set_label_coords(1.05, 0.175)

    #--> Pas de ticks le long de l'axe y.
    ax.set_yticks([], [])

    #--> Remplace le cadre habituel de la figure par un systeme d'axes
    #    centre en (0, 0).
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))

    #--> Ajout des fleches au bout des axes.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin)
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=0.25*hl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    # --> Save figure.
    plt.savefig('../Slides/imgs/Newton_method.pdf', bbox_inches='tight', dpi=300)
    plt.show()
