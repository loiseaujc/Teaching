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

def jacobian_matrix(x, params):

    # --> Unpack parameters.
    sigma, rho, beta = params

    # --> Initialize variables.
    A = np.zeros((x.size, x.size))

    # --> x-equation.
    A[0, 0], A[0, 1] = -sigma, sigma

    # --> y-equation.
    A[1, 0], A[1, 1], A[1, 2] = rho - x[2], -1, -x[0]

    # --> z-equation.
    A[2, 0], A[2, 1], A[2, 2] = x[1], x[0], -beta

    return A

if __name__ == '__main__':

    from scipy.linalg import eig
    from scipy.integrate import odeint

    # --> Conducting state.
    xc = np.zeros((3,))

    #######################################
    #####                             #####
    #####     PRIMARY BIFURCATION     #####
    #####                             #####
    #######################################

    # --> Sets the parameters.
    rho = [1.1, 2.5, 5., 10., 13.9]
    fig = plt.figure(figsize=(w, w/3))
    for i, r in enumerate(rho):
        ax = plt.subplot(1, 5, i+1, projection='3d')

        params = [10, r, 8./3.]

        # --> Compute the Jacobian matrix for the conducting state.
        A = jacobian_matrix(xc, params)
        eigenvalues, eigenvectors = eig(A)
        idx = np.argsort(-eigenvalues.real)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        # --> Plot figure.
        # fig = plt.figure(figsize=(w/2, w/2))
        # ax = fig.gca(projection='3d')

        # --> Integrate initial condition.
        t = np.linspace(0, 1000, 100000)
        x0 = 1e-3*eigenvectors[:, 0].real
        x = odeint(lorenz_system, x0, t, args=(params,))
        ax.plot(x[:, 0], x[:, 1], x[:, 2], color='royalblue')
        x = odeint(lorenz_system, -x0, t, args=(params,))
        ax.plot(x[:, 0], x[:, 1], x[:, 2], color='orange')
        ax.axis('off')
        ax.set_title(r'$\rho = %.2f$' %r)

    plt.savefig('../imgs/primary_bifurcation_phase_plots.pdf', bbox_inches='tight', dpi=300)


    #####
    #####
    #####     HOMOCLINIC CONNECTION     #####
    #####
    #####

    params = [10, 13.926, 8./3.]

    # --> Compute the Jacobian matrix for the conducting state.
    A = jacobian_matrix(xc, params)
    eigenvalues, eigenvectors = eig(A)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    # --> Plot figure.
    fig = plt.figure(figsize=(w/2, w/2))
    ax = fig.gca(projection='3d')

    # --> Integrate initial condition.
    t = np.linspace(0, 6, 100000)
    x0 = 1e-11*eigenvectors[:, 0].real
    x = odeint(lorenz_system, x0, t, args=(params,))
    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='royalblue')
    x = odeint(lorenz_system, -x0, t, args=(params,))
    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='orange')
    ax.axis('off')
    ax.set_title(r'$\rho = 13.926$')

    plt.savefig('../imgs/homoclinic_connection.pdf', bbox_inches='tight', dpi=300)





    #####
    #####
    #####     PRE-CHAOTIC TRANSIENTS
    #####
    #####

    # params = [10., 21., 8./3.]
    #
    # N_trajectories = 4
    # colors = plt.cm.viridis(np.linspace(0, 1, N_trajectories))
    #
    # fig = plt.figure(figsize=(w, w/4))
    # fig_2 = plt.figure(figsize=(w/2, w/2))
    # ax = fig.gca()
    # ax_2 = fig_2.gca(projection='3d')
    #
    # t = np.linspace(0, 100, 10000)
    # from numpy.random import uniform
    #
    # for i in xrange(N_trajectories):
    #     x0 = uniform(low=-15, high=15, size=(3,))
    #     x = odeint(lorenz_system, x0, t, args=(params,))
    #     ax.plot(t, x[:, 0], color=colors[i])
    #
    #     ax_2.plot(x[:, 0], x[:, 1], x[:, 2], color=colors[i])
    #
    # ax_2.axis('off')
    #
    # ax.set_xlabel(r'$t$')
    # ax.set_ylabel(r'$x$')
    #
    # fig.savefig('../imgs/chaotic_transients_time_series.pdf', bbox_inches='tight', dpi=300)
    # fig_2.savefig('../imgs/chaotic_transients_phase_space.pdf', bbox_inches='tight', dpi=300)




    #####
    #####
    #####     SUBCRITICAL HOPF BIFURCATION
    #####
    #####

    N = 7
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    rho = np.arange(22, 22+N)

    fig = plt.figure(figsize=(w/2, w/4))
    ax = fig.gca()
    for i, r in enumerate(rho):
        params = [10., r, 8./3.]
        x = np.array([np.sqrt(-params[-1] + params[-1]*params[1]),
                      np.sqrt(-params[-1] + params[-1]*params[1]),
                      -1 + params[1]
                      ])
        A = jacobian_matrix(x, params)
        eigenvalues, _ = eig(A)

        ax.plot(r, eigenvalues.real.max(), 'o', color=colors[i])

        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(21, 29)
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$\Re(\lambda)$')

    plt.savefig('../imgs/hopf_bifurcation_eigenvalue.pdf', bbox_inches='tight', dpi=300)



    #####
    #####
    #####     CHAOTIC DYNAMICS
    #####
    #####

    params = [10., 28., 8./3.]
    x0 = np.array([1., 1., 1.])
    t = np.linspace(0, 250, 25000)
    x = odeint(lorenz_system, x0, t, args=(params,))

    x0 = x[-1, :]
    x = odeint(lorenz_system, x0, t, args=(params,))

    fig = plt.figure(figsize=(w/2, w/2))
    ax = fig.gca(projection='3d')
    ax.plot(x[-5000:, 0], x[-5000:, 1], x[-5000:, 2], color='royalblue')
    ax.axis('off')

    plt.savefig('../imgs/strange_attractor.pdf', bbox_inches='tight', dpi=300)

    y = odeint(lorenz_system, x0+1e-8, t, args=(params,))
    fig = plt.figure(figsize=(w, w/3))
    ax = fig.gca()
    ax.plot(t, x[:, 0], color='royalblue')
    ax.plot(t, y[:, 0], color='orange', ls='--')

    ax.set_xlim(0, 50)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')

    plt.savefig('../imgs/sensitivity_initial_conditions.pdf', bbox_inches='tight', dpi=300)

    fig = plt.figure(figsize=(w/2, w/2))
    ax = fig.gca()

    r = abs(x[:, -1] - y[:, -1])

    ax.semilogy(t, r, color='royalblue')
    ax.set_xlim(0, 50)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\vert z_1 - z_2 \vert$')

    plt.savefig('../imgs/sensitivity_initial_conditions_bis.pdf', bbox_inches='tight', dpi=300)


    plt.show()
