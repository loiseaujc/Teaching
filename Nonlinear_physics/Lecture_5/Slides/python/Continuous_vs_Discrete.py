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

    t = np.linspace(-2*np.pi, 2*np.pi, 2*80+1)
    x = np.cos(2*t)*np.cos(t)

    # --> Continuous vs. Discrete time.

    fig, axes = plt.subplots(1, 2, figsize=(w, w/4), sharex=True, sharey=True)
    axes[0].plot(t, x, color='royalblue')
    axes[0].set_ylabel(r'$x(t)$')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_title(r'(a) Continuous time', y=-0.55)

    axes[1].plot(t[::8], x[::8], '.', color='royalblue')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_title(r'(b) Discrete time', y=-0.55)
    axes[1].set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/Continuous_vs_Discrete.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(w, w/4), sharex=True, sharey=True)
    axes[0].plot(t, x, color='royalblue')
    axes[0].plot(t, x.mean()*np.ones_like(x), color='dimgrey', ls='--')
    axes[0].set_ylabel(r'$x(t)$')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_title(r'(a) Continuous time', y=-0.55)

    axes[1].plot(t[::8], x[::8], '.', color='royalblue')
    axes[1].plot(t, x.mean()*np.ones_like(x), color='dimgrey', ls='--')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_title(r'(b) Discrete time', y=-0.55)
    axes[1].set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/Continuous_vs_Discrete_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # --> Cardinal sinus.
    xx = np.linspace(-3*np.pi, 3*np.pi, 2000)
    y = np.sin(np.pi*xx)/(np.pi*xx)
    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(xx, y, color='royalblue')
    ax.set_ylabel(r'$sinc(t)$')
    ax.set_xlabel(r'$t$')
    # ax.set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/cardinal_sinus.pdf', bbox_inches='tight', dpi=300)

    # --> Sampling theorem.
    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t, x, color='royalblue')
    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/Sampling_theorem_a.pdf', bbox_inches='tight', dpi=300)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t, x, color='royalblue')
    ax.plot(t[::8], x[::8], '.', color='red')
    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/Sampling_theorem_b.pdf', bbox_inches='tight', dpi=300)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t[::8], x[::8], '.', color='red')
    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/Sampling_theorem_c.pdf', bbox_inches='tight', dpi=300)

    that, xhat = t[::8], x[::8]
    Ts = that[1]-that[0]
    t_ = np.linspace(-np.pi, np.pi, 1000)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    for i in xrange(-10, 11):
        if i == 0:
            ax.plot(t_, xhat[i+10]*np.sinc((t_-i*Ts)/Ts), color='lightgrey')

    ax.plot(that, xhat, '.', color='red')
    ax.set_xlim(-np.pi, np.pi)

    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel(r'$t$')

    plt.savefig('../imgs/Sampling_theorem_d.pdf', bbox_inches='tight', dpi=300)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    for i in xrange(-10, 11):
        ax.plot(t_, xhat[i+10]*np.sinc((t_-i*Ts)/Ts), color='lightgrey')
    ax.plot(that, xhat, '.', color='red')

    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_xlim(-np.pi, np.pi)

    plt.savefig('../imgs/Sampling_theorem_e.pdf', bbox_inches='tight', dpi=300)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    f = np.zeros_like(t_)
    for i in xrange(-10, 11):
        f += xhat[i+10]*np.sinc((t_-i*Ts)/Ts)
        ax.plot(t_, xhat[i+10]*np.sinc((t_-i*Ts)/Ts), color='lightgrey')
    ax.plot(t_, f, color='royalblue')
    ax.plot(that, xhat, '.', color='red')
    ax.set_xlim(-np.pi, np.pi)

    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel(r'$t$')

    plt.savefig('../imgs/Sampling_theorem_f.pdf', bbox_inches='tight', dpi=300)

    plt.show()
