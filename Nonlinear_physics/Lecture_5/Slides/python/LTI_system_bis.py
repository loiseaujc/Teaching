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

    # --> First-order system.
    k = -0.1
    t = np.linspace(0, 100)

    A, B, C, D = np.array([k])[:, np.newaxis], np.array([1])[:, np.newaxis], np.array([1])[:, np.newaxis], np.array([0])[:, np.newaxis]
    from scipy.signal import lti, lsim2
    LTI = lti(A, B, C, D)
    _, y, _ = lsim2(LTI, X0=1., T=t)

    from scipy.signal import cont2discrete
    Ts = t[5]-t[0]
    DLTI = cont2discrete((A, B, C, D), Ts)

    from scipy.signal import dlti, dlsim
    DLTI = dlti(DLTI[0], DLTI[1], DLTI[2], DLTI[3], dt=DLTI[4])
    td, yd, _ = dlsim(DLTI, t[::5]*0, x0=1)

    fig = plt.figure(figsize=(w/3, w/3))
    ax = fig.gca()
    ax.plot(t, y)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../imgs/first_order_system.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(w/3, w/3))
    ax = fig.gca()
    ax.plot(t, y)
    ax.plot(td, yd, '.')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../imgs/first_order_system_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # --> Second-order system.
    t = np.linspace(0, 100, 1001)
    dt = t[1]-t[0]
    k, omega = 0.1, 1.
    A = np.array([[0, 1], [-omega**2, -2*k]])
    B = np.array([[0], [1]])
    C = np.array([1, 0])
    D = np.array([0])

    x0 = np.array([1, 0])
    LTI = lti(A, B, C, D)
    _, y, _ = lsim2(LTI, X0=x0, T=t)

    Ts = np.pi/2.
    lag = int(Ts/(t[1]-t[0]))
    DLTI = cont2discrete((A, B, C, D), Ts)
    DLTI = dlti(DLTI[0], DLTI[1], DLTI[2], DLTI[3], dt=DLTI[4])
    td, yd, _ = dlsim(DLTI, t[::lag]*0, x0=x0)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t, y)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../imgs/second_order_system.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(t, y)
    ax.plot(td[:-3], yd[:-3], '.')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    plt.savefig('../imgs/second_order_system_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # --> Impulse responses.
    from scipy.signal import impulse, dimpulse, step, dstep
    tc, hc = impulse(LTI, T=t)
    td, hd = dimpulse(DLTI, t=td)

    fig = plt.figure(figsize=(w, w/4))
    ax = fig.gca()
    ax.plot(tc, hc, label=r'Continuous')
    ax.plot(td, hd[0], '.', label=r'Discrete')
    ax.legend(loc=0)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$h(t)$, $h[n]$')
    ax.set_ylim(-1.2, 1.2)

    plt.savefig('../imgs/impulse_response.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # --> Response to generic forcing.
    from scipy.signal import convolve
    u = np.cos(0.5*np.pi*tc)
    yc = convolve(hc, u, mode='full')[:tc.size]*dt
    x0 = np.array([0, 0])
    _, ycc, _ = lsim2(LTI, X0=x0, T=tc, U=u)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(w, w/3))
    axes[0].plot(tc, u)
    axes[0].set_ylabel(r'$u(t)$')

    axes[1].plot(tc, ycc, label=r'Simulation')
    axes[1].plot(tc, yc, ls='--', label=r'Convolution')
    axes[1].set_ylabel(r'$y(t)$')
    axes[1].set_xlabel(r'$t$')
    axes[1].legend(loc=0)

    plt.savefig('../imgs/LTI_simulation.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(w/2, w/4))
    ax = fig.gca()
    ax.plot(tc, u, label=r'Input')
    ax.plot(tc, yc, label=r'Output')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
    ax.set_xlim(80, 100)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$u(t)$, $y(t)$')

    plt.savefig('../imgs/input_output_shift.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # --> Step response.
    tc, sc = step(LTI, T=t)

    tc = np.concatenate((-np.flipud(tc), tc), axis=0)
    uc = np.zeros_like(tc)
    uc[tc.size/2:] = 1.
    sc = np.concatenate((0*sc, sc), axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(w/2, w/2), sharex=True)
    axes[0].plot(tc, uc)
    axes[0].set_ylabel(r'$u(t)$')

    axes[1].plot(tc, sc)
    axes[1].set_ylabel(r'$y(t)$')
    axes[1].set_xlabel(r'$t$')

    plt.savefig('../imgs/unit_step_response.pdf', bbox_inches='tight', dpi=300)
    plt.show()
