import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from scipy.linalg import solve

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams.update(params)
colors = [ 'dimgrey', 'royalblue', 'orange', 'seagreen', 'y' ]

fig_width = 5.33

def streamlines(x, y, u, v, xf, yf, density=1, savename=None):

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    #-->
    magnitude = np.sqrt(u**2 + v**2)

    ax.streamplot(x, y, u, v, color=magnitude, cmap=plt.cm.inferno, density=density)
    ax.plot(xf, yf, 'ro')

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    if savename is not None:
        plt.savefig(savename, bbox_inches='tight', dpi=300)

    return

def dynamical_system(x):

    # --> Initialize variables.
    dx = np.zeros_like(x)

    # --> x-equation.
    dx[0] = x[0] - x[1]**2 + 1.28 + 1.4 *x[0]*x[1]

    # --> y-equation.
    dx[1] = 0.2*x[1] - x[0] + x[0]**3

    return dx

def jacobian_matrix(x):

    # --> Initialize variable.
    J = np.zeros((x.shape[0], x.shape[0]))

    # --> Jacobian matrix.
    J[0, 0] = 1. + 1.4*x[1]
    J[0, 1] = -2.*x[1] + 1.4*x[0]
    J[1, 0] = -1. + 3*x[0]**2.
    J[1, 1] = 0.2

    return J

def LTI_system(A, x):
    dx = np.zeros_like(x)
    dx[0] = A[0, 0]*x[0] + A[0, 1]*x[1]
    dx[1] = A[1, 0]*x[0] + A[1, 1]*x[1]
    return dx

def newton(x):

    for i in xrange(10):
        # --> Evaluate function and jacobian.
        f = dynamical_system(x)
        J = jacobian_matrix(x)

        # --> Solve linear system.
        d = solve(J, -f)

        # --> Update solution.
        x += d

    return x

if __name__ == '__main__':

    from scipy.linalg import eig

    # --> Definition of the grid for the phase plane.
    x = np.linspace(-3, 3, 100);
    x, y = np.meshgrid(x, x);

    # --> Compute the time derivative vectors.
    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = dynamical_system([x[:], y[:]])

    # --> First fixed points.
    x0 = np.array([0., -1.])
    x1 = newton(x0)

    J = jacobian_matrix(x1)
    eigenvalues, eigevnvectors = eig(J)

    xdot_p, ydot_p = np.zeros_like(xdot), np.zeros_like(ydot)
    xdot_p[:], ydot_p[:] = LTI_system(J, [x[:], y[:]])

    streamlines(x, y, xdot, ydot, x1[0], x1[1], savename='../Slides/imgs/fixed_points_2.pdf')
    streamlines(x, y, xdot_p, ydot_p, 0, 0, savename='../Slides/imgs/fixed_points_2_bis.pdf')

    print 'The first fixed point is given by :', x1
    print 'Its eigenvalues are :', eigenvalues

    plt.show()
