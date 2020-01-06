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

def phase_line_plot(x, f):

    """

    Trace l'espace des phases pour un systeme dynamique du premier ordre.

    INPUTS:
    ------

    x : Discretisation de l'espace des phases 1D.
        one-dimensional numpy array.

    f : Fonction f(x) caracterisant le systeme dynamique.

    """

    #---> Creation de la figure.
    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    #--> Trace la courbe dx/dt = f(x).
    ax.plot(x, f, color=colors[0])
    #--> Place les points d'equilibre.
    eq_stab = np.arange(-np.pi, 2*np.pi, 2*np.pi)
    eq_unstab = np.arange(-2*np.pi, 3*np.pi, 2*np.pi)
    ax.plot(eq_stab, 0.*eq_stab, 'o', ms=4, color=colors[1])
    ax.plot(eq_unstab, 0.*eq_unstab, 's', ms=4, color=colors[2])

    #-------------------------------------#
    #----- Mise en page de la figure -----#
    #-------------------------------------#

    #--> Bornes inferieure et superieure pour l'axe des x.
    ax.set_xlim(-8, 8)
    ax.set_ylim(-1.25, 1.25)
    #--> Label de l'axe des x.
    ax.set_xlabel(r'$x$')
    #--> Positionnement de ce label sur la figure.
    ax.xaxis.set_label_coords(1.05, 0.5)
    #--> Choix des ticks et de leur label le long de l'axe x.
    ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$'])
    #--> Petit carre semi-transparent autour des graduations.
    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    #--> Label de l'axe des y.
    ax.set_ylabel(r'$\dot{x}$', rotation=0)
    #--> Positionnement de ce label.
    ax.yaxis.set_label_coords(0.5, 1.05)
    #--> Pas de ticks le long de l'axe y.
    ax.set_yticks([], [])

    #--> Remplace le cadre habituel de la figure par un systeme d'axes
    #    centre en (0, 0).
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

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

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=0.25*yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    return

if __name__ == '__main__':

    # --> Define the function.
    f = lambda x : np.sin(x)

    # --> Define the mesh.
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)

    phase_line_plot(x, f(x))

    plt.savefig('../Slides/imgs/flow_on_the_line.pdf', bbox_inches='tight', dpi=300)

    # --> Simulate the evolution of the system.
    def dynamical_system(x, t):

        dx = np.sin(x)

        return dx

    from scipy.integrate import odeint
    t = np.linspace(0, 30, 1000)
    x0 = 1e-6

    x = odeint(dynamical_system, x0, t)

    # --> Plot the figure.
    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    ax.plot(t, x, color=colors[1])

    # Set y ticks.
    ax.set_yticks([0., np.pi/2, np.pi])
    ax.set_yticklabels([r'~', r'$\displaystyle \frac{\pi}{2}$', r'$\pi$'])
    ax.set_ylim(-0.1, 1.1*np.pi)


    #--> Remplace le cadre habituel de la figure par un systeme d'axes
    #    centre en (0, 0).
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', -0.1))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

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
    ax.arrow(xmin, -0.1, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=0.25*hl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=0.25*yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../Slides/imgs/flow_on_the_line_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()
