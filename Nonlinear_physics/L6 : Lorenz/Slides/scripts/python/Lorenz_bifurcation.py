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

def lorenz_system(x, t, params):

    # --> Unpack parameters.
    sigma, rho, beta = params

    # --> Initialize variables.
    dx = np.zeros_like(x)

    # --> x-equation.
    dx[0] = sigma*(x[1] - x[0])
    # --> y-equation.
    dx[1] = x[0]*(rho - x[2]) - x[1]
    # --> z-equation.
    dx[2] = x[0]*x[1] - beta*x[2]

    return dx

if __name__ == '__main__':

    # --> Miscellaneous.
    from scipy.signal import find_peaks_cwt
    from scipy.integrate import odeint
    from detect_peak import *


    rho = np.linspace(0, 35, 351)

    fig = plt.figure(figsize=(w, w/3))
    ax = fig.gca()

    for r in rho:
        print r

        if r < 2:
            x0 = np.array([0., 1., 0.])

        params = [10., r, 8./3.]
        t = np.linspace(0, 1000, 10000)
        x = odeint(lorenz_system, x0, t, args=(params,))
        x0 = x[-1, :]
        x = x[-5000:, 0]
        peakind = detect_peaks(x)
        if peakind.size == 0:
            z = x[-1]
        else:
            z = x[peakind]
        ax.plot(r*np.ones_like(z), z, '.', color='royalblue', ms=1)

        if r < 2:
            y0 = np.array([0., -2., 0.])

        y = odeint(lorenz_system, y0, t, args=(params,))
        y0 = y[-1, :]
        y = y[-5000:, 0]
        peakind = detect_peaks(y)
        if peakind.size == 0:
            z = y[-1]
        else:
            z = y[peakind]
        ax.plot(r*np.ones_like(z), z, '.', color='royalblue', ms=1)

    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$x_{\max}$')

    plt.savefig('../imgs/lorenz_bifurcation_diagram_1.pdf', bbox_inches='tight', dpi=300)

    rho = np.linspace(0, 500, 501)

    fig = plt.figure(figsize=(w, w/3))
    ax = fig.gca()

    for r in rho:
        print r

        if r < 2:
            x0 = np.array([0., 1., 0.])

        params = [10., r, 8./3.]
        t = np.linspace(0, 1000, 500000)
        x = odeint(lorenz_system, x0, t, args=(params,))
        x0 = x[-1, :]
        x = x[-50000:, 0]
        peakind = detect_peaks(x)
        if peakind.size == 0:
            z = x[-1]
        else:
            z = x[peakind]
        ax.plot(r*np.ones_like(z), z, '.', color='royalblue', ms=1)

        if r < 2:
            y0 = np.array([0., -2., 0.])

        y = odeint(lorenz_system, y0, t, args=(params,))
        y0 = y[-1, :]
        y = y[-50000:, 0]
        peakind = detect_peaks(y)
        if peakind.size == 0:
            z = y[-1]
        else:
            z = y[peakind]
        ax.plot(r*np.ones_like(z), z, '.', color='royalblue', ms=1)

    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$x_{\max}$')

    plt.savefig('../imgs/lorenz_bifurcation_diagram_2.pdf', bbox_inches='tight', dpi=300)

    plt.show()
