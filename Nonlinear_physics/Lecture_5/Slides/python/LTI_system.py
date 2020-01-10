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

    from scipy.integrate import odeint

    # --> First-order continuous time LTI system.
    t = np.linspace(0, 100)
    k = 0.1

    def lti_system(x, t):
        dx = -k * x
        return dx

    x0 = 1.
    x = odeint(lti_system, x0, t)

    fig = plt.figure(figsize=(w/3, w/3))
    ax = fig.gca()
    ax.plot(t, x, color='royalblue')
    ax.plot(t[0], x[0], '.', color='red')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../imgs/first_order_system.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    alpha = np.exp(-k*t[5])
    y = np.zeros((10, ))
    y[0] = x0
    for i in xrange(9):
        y[i+1] = alpha*y[i]

    fig = plt.figure(figsize=(w/3, w/3))
    ax = fig.gca()
    ax.plot(t, x, color='royalblue', label=r'Continuous time')
    ax.plot(t[::5], y, '.', color='red', label=r'Discrete time')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35))

    plt.savefig('../imgs/first_order_system_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # --> First-order continuous time LTI system.
    t = np.linspace(0, 50, 1000)
    k, omega = 0.1, 1

    def lti_system(x, t):
        # --> Initialize variables.
        dx = np.zeros_like(x)
        # --> x-equation.
        dx[0] = x[1]
        # --> y-equation.
        dx[1] = -2*k * x[1] - omega**2*x[0]
        return dx

    x0 = np.array([1., 0.])
    x = odeint(lti_system, x0, t)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t, x[:, 0], color='royalblue')
    ax.plot(t[0], x[0, 0], '.', color='red')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../imgs/second_order_system.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    lag = int(np.pi/2. / (t[1]-t[0]))
    y = np.roll(x[:, 0], -2*lag, axis=0)[:-2*lag]
    x1 = np.roll(x[:, 0], -lag, axis=0)[:-2*lag]
    x2 = x[:-2*lag, 0]

    X = np.concatenate((x1[:, np.newaxis], x2[:, np.newaxis]), axis=1)
    coef = np.linalg.lstsq(X, y)[0]

    y = np.zeros_like(x[::lag, 0])
    y[0] = x[0, 0]
    y[1] = x[lag, 0]
    for i in xrange(2, y.size):
        y[i] = coef[0]*y[i-1] + coef[1]*y[i-2]

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t, x[:, 0], color='royalblue', label=r'Continuous time')
    ax.plot(t[::lag], y, '.', color='red', label=r'Discrete time')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')
    ax.legend(loc=0)

    plt.savefig('../imgs/second_order_system_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    from scipy.signal import dlti
    A = np.array([[coef[0], coef[1]], [1., 0.]])
    B = np.array([[1], [0]])
    C = np.array([1, 0])
    D = np.array([0])
