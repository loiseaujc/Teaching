# --> Import standard python library.
import numpy as np

# --> Import and setup matplotlib for figures.
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'text.usetex' : True,
          'font.family' : 'lmodern',
          'axes.titlepad': 10}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams.update(params)

# --> Import FFT package from scipy.
from scipy.fftpack import fft2, ifft2, fftfreq

from tqdm import trange
from celluloid import Camera
import cmasher as cmr

# --> Define the class NS2D.

class NS2D:
    '''
    This class implements a strictly 2D Fourier spectral Navier-Stokes solver.
    It is based on the matlab implementation by Kunihiko (Sam) Taira (Florida State
    University, www.eng.fsu.edu/~ktaira).
    '''

    def __init__(self, lx=2*np.pi, ly=2.*np.pi, nu=5e-4, nx=128, ny=128, T=10., iostep=10, sparsity=None, antialiasing=False):
        '''
        This function initializes the NS2D class.
        '''

        ###########################################
        #####     USER-DEFINED QUANTITIES     #####
        ###########################################

        # --> Streamwise extent of the computational domain.
        self.lx = lx
        # --> Spanwise extent of the computational domain.
        self.ly = ly
        # --> Number of grid points in the x-direction.
        self.nx = nx
        # --> Number of grid points in the y-direction.
        self.ny = ny
        # --> Maximum integration time.
        self.T = T
        # --> Input/Output step.
        self.iostep = iostep
        # --> Sparsification.
        self.sparsity = sparsity
        # --> Anti-aliasing.
        self.antialiasing = antialiasing
        # --> Viscosity (Reynolds number)
        self.nu = nu

        ######################################
        #####     DERIVED QUANTITIES     #####
        ######################################

        # --> Grid-spacing in the x-direction.
        self.dx = lx/nx
        # --> Grid-spacing in the y-direction.
        self.dy = ly/ny
        # --> Time-step used for the simulation.
        self.dt = 0.25*self.dx # NOTE: Suggested by Sam Taira.
        # --> Number of time-steps to be performed.
        self.nsteps = np.ceil(T/self.dt).astype('int')
        print(self.nsteps)
        # --> Define the grid in the x-direction.
        x = np.linspace(0, lx, nx+1)
        # --> Define the grid in the y-direction.
        y = np.linspace(0, ly, ny+1)
        # --> Define the 2D grid for plotting purposes.
        self.xp, self.yp = np.meshgrid(x, y)
        # --> Define the 2D grid for computation purposes.
        self.xx, self.yy = np.meshgrid(x[:-1], y[:-1]) # NOTE: Last point is removed for coding (periodicity implied)
        # --> Sets-up the number of points for dealiasing in the spectral space.
        if np.mod(nx, 2) + np.mod(ny, 2) > 0:
            raise ValueError('nx and ny must be divisible by 2.')
        else:
            self.nxp = int(3*nx/2)
            self.nyp = int(3*ny/2)
        # --> Define the Fourier wavenumbers in the x-direction.
        dx, dxp = self.lx/(2*np.pi*self.nx), self.lx/(2*np.pi*self.nxp)
        self.kx, self.kxp = fftfreq(self.nx, d=dx), fftfreq(self.nxp, d=dxp)
        # self.kx = 2*np.pi * np.concatenate((np.arange(0, nx/2 + 1), np.arange(-nx/2+1, 0))) / lx
        # self.kxp = 2*np.pi * np.concatenate((np.arange(0, self.nxp/2 + 1), np.arange(-self.nxp/2+1, 0))) / lx # NOTE: Dealiasing purposes.
        # --> Define the Fourier wavenumbers in the y-direction.
        dy, dyp = self.ly/(2*np.pi*self.ny), self.ly/(2*np.pi*self.nyp)
        self.ky, self.kyp = fftfreq(self.ny, d=dy), fftfreq(self.nyp, d=dyp)
        # self.ky = 2*np.pi * np.concatenate((np.arange(0, ny/2 + 1), np.arange(-ny/2+1, 0))) / ly
        # self.kyp = 2*np.pi * np.concatenate((np.arange(0, self.nyp/2 + 1), np.arange(-self.nyp/2+1, 0))) / ly # NOTE: Dealiasing purposes.

        self.k2 = self.kx[:, np.newaxis]**2 + self.ky[np.newaxis, :]**2
        self.k2[0, 0] = 1e-9

        #################################
        #####     MISCELLANEOUS     #####
        #################################

        # --> Current time step.
        self.istep = 0
        # --> Counter for plot.
        self.counter = 0

    def initial_condition(self, case=3, umax=1., random_state=None, nmax=100):
        '''
        This function defines the initial condition for the simulation of
        two-dimensional Navier-Stokes turbulence. The initial condition is
        made by a random combination of Taylor vortices.
        '''

        if case == 1:
            # --> Single Taylor vortex.
            omg_hat = self.taylor_vortex(x0=0.5*self.lx, y0=0.5*self.ly, r=self.lx/8., umax=umax)

        elif case == 2:
            # --> Two co-rotating Taylor vortices.
            omg_hat = self.taylor_vortex(x0=0.5*self.lx, y0=0.4*self.ly, r=0.1*self.lx, umax=1.)
            omg_hat += self.taylor_vortex(x0=0.5*self.lx, y0=0.6*self.ly, r=0.1*self.lx, umax=1.)

        elif case == 3:
            # --> Sets-up the seed of random number generator for reproductibility.
            if random_state is not None:
                from numpy.random import seed
                seed(seed=random_state)

            # --> Sets-up the distribution of Taylor vortices.
            from numpy.random import uniform, normal, gamma
            loc = uniform(low=0., high=self.lx, size=(2,))
            umax = normal(loc=0, scale=1, size=(1,))
            radius = self.lx / 20.
            omg_hat = self.taylor_vortex(x0=loc[0]*self.lx, y0=loc[1]*self.ly, r=radius, umax=umax)

            for i in range(nmax):
                loc = uniform(low=-0.5, high=0.5, size=(2,))
                umax = normal(loc=0, scale=1, size=(1,))
                omg_hat += self.taylor_vortex(x0=loc[0]*self.lx, y0=loc[1]*self.ly, r=radius, umax=umax)

        return omg_hat

    def taylor_vortex(self, x0=0., y0=0., r=0.125, umax=1.):
        '''
        This function computes the vorticity distribution of a single Taylor
        vortex.
        '''

        # --> Initial the vorticity distribution in the physical space.
        omega = np.zeros_like(self.xx)
        # --> Define the vorticity distribution in physical space.
        # TODO: Find some references about that.
        for i in range(-1, 2):
            for j in range(-1, 2):
                r2 = (self.xx - x0 - i*self.lx)**2 + (self.yy - y0 - j*self.ly)**2
                omega += (umax/r) * (2. - r2/r**2) * np.exp(0.5*(1. - r2/r**2))

        # --> Compute the corresponding vorticity distribution in the
        #     spectral space.
        omega_hat = fft2(omega)

        return omega_hat

    def plot_figure(self, fig, omega_hat):
        '''
        This function takes as input a given vorticity field expressed in the
        spectral space and outposts a publication-ready figure of this field
        in both the spectral space and the physical space.
        '''

        #####################################################
        #####     Vorticity field in physical space     #####
        #####################################################

        # --> Sets-up the figure and its axis.
        ax = fig.gca()

        # --> Computes vorticity field in the physical space.
        omega = ifft2(omega_hat).real
        amp = max(abs(omega).max(), abs(omega).min())
        amp *= 0.5

        # --> Computes velocity in physical space.
        u, v, _ = self._compute_velocity(omega_hat)

        omega_p = np.zeros_like(self.xp)
        omega_p[:-1, :-1] = omega
        omega_p[-1, :] = omega_p[0, :]
        omega_p[:, -1] = omega_p[:, 0]

        u_p = np.zeros_like(self.xp)
        u_p[:-1, :-1] = u
        u_p[-1, :] = u_p[0, :]
        u_p[:, -1] = u_p[:, 0]

        v_p = np.zeros_like(self.xp)
        v_p[:-1, :-1] = v
        v_p[-1, :] = v_p[0, :]
        v_p[:, -1] = v_p[:, 0]

        if omega.min() * omega.max() < 0:
            ax.pcolormesh(self.xp, self.yp, omega_p, vmin=-amp, vmax=amp, shading='gouraud', cmap=cmr.iceburn)
            # ax.quiver(self.xp[::4, ::4], self.yp[::4, ::4], u_p[::4, ::4], v_p[::4, ::4])
        else:
            if abs(omega).max() < abs(omega).min():
                ax.pcolormesh(self.xp, self.yp, omega_p, vmin=-amp, vmax=0, shading='gouraud', cmap=plt.cm.hot_r)
            else:
                ax.pcolormesh(self.xp, self.yp, omega_p, vmin=0, vmax=amp, shading='gouraud', cmap=plt.cm.hot_r)
        ax.set_aspect('equal')

        # --> Decorators.
        # ax.locator_params(axis='both', nbins=5)
        # ax.set_xlabel(r'$x$')
        # ax.set_ylabel(r'$y$')
        ax.set_facecolor("black")

        ax.axis("off")
        plt.tight_layout()

    def _advection(self, omega_hat):
        '''
        This function computes the nonlinear advection using a pseudo-spectral
        approach, with or without de-aliasing.
        '''

        # --> Initialize various variables.
        # uhat = np.zeros_like(omega_hat)         # x-velocity
        # vhat = np.zeros_like(omega_hat)         # y-velocity
        #
        # psi_hat = np.zeros_like(omega_hat)      # Streamfunction.
        #
        # domgdx = np.zeros_like(omega_hat)       # x-gradient of vorticity.
        # domgdy = np.zeros_like(omega_hat)       # y-gradient of vorticity.

        # # --> Computation of the unpadded functions.
        # for i in range(self.ny):
        #     for j in range(self.nx):
        #         # --> Solve for the stream function first. (No padding needed).
        #         if i == 0 and j == 0:
        #             # NOTE: For alpha = beta = 0.
        #             psi_hat[i, j] = 0
        #         else:
        #             k2 = self.kx[j]**2 + self.ky[i]**2
        #             psi_hat[i, j] = omega_hat[i, j] / k2
        #
        #         # --> Unpadded gradient of vorticity for advection in spectral space.
        #         domgdx[i, j] = 1j*self.kx[j]*omega_hat[i, j]
        #         domgdy[i, j] = 1j*self.ky[i]*omega_hat[i, j]
        #
        #         # --> Unpadded velocity in spectral space.
        #         uhat[i, j] = 1j*self.ky[i]*psi_hat[i, j]
        #         vhat[i, j] = -1j*self.kx[j]*psi_hat[i, j]

        psi_hat = omega_hat/self.k2
        domgdx = 1j*self.kx[np.newaxis, :]*omega_hat
        domgdy = 1j*self.ky[:, np.newaxis]*omega_hat
        uhat = 1j*self.ky[:, np.newaxis]*psi_hat
        vhat = -1j*self.kx[np.newaxis, :]*psi_hat

        if self.antialiasing is False:
            # NOTE: No anti-aliasing is performed in this case.
            u = ifft2(uhat)
            v = ifft2(vhat)
            adv = fft2(-u*ifft2(domgdx) - v*ifft2(domgdy))
        else:
            up = ifft2(self._pad(uhat))
            vp = ifft2(self._pad(vhat))

            domgdxp = ifft2(self._pad(domgdx))
            domgdyp = ifft2(self._pad(domgdy))

            adv = -self._chop(fft2(up*domgdxp + vp*domgdyp))*(1.5**2)

        return adv

    def _rhs(self, omega_hat):
        '''
        This function computes the right-hand side of the Navier-Stokes
        equations in streamfunction-vorticity formulation.
        '''

        # --> Computation of the diffusion term.
        # diffusion = np.zeros_like(omega_hat)
        # for i in range(self.ny):
        #     for j in range(self.nx):
        #         k2 = self.kx[j]**2 + self.ky[i]**2
        #         diffusion[i, j] = -self.nu*k2*omega_hat[i, j]

        diffusion = -self.nu*self.k2*omega_hat

        # --> Computation of the advection term.
        advection = self._advection(omega_hat)

        # --> Computation of the right-hand side.
        rhs = advection + diffusion
        rhs[0, 0] = 0

        return rhs

    def _rk4(self, x):
        '''
        Fourth-order accurate Runge-Kutta method.
        '''

        k1 = self._rhs(x)
        k2 = self._rhs(x + 0.5*self.dt*k1)
        k3 = self._rhs(x + 0.5*self.dt*k2)
        k4 = self._rhs(x + self.dt*k3)

        x += self.dt * (k1 + 2.*( k2 + k3 ) + k4)/6.

        return x

    def _pad(self, x):
        '''
        This function pads the vorticity field in the wavespace to store spurious
        high frequency components resulting from the computation of the nonlinear
        term in the physical space.
        '''

        # --> Initialize variable.
        xp = np.zeros((self.nxp, self.nyp)).astype('complex128')

        # --> Input the original vorticity field.
        xp[0:self.ny//2, 0:self.nx//2] = x[0:self.ny//2, 0:self.nx//2]
        xp[0:self.ny//2, -self.nx//2+1:] = x[0:self.ny//2, self.nx//2+1:]
        xp[-self.ny//2+1:, 0:self.nx//2] = x[self.ny//2+1:, 0:self.nx//2]
        xp[-self.ny//2+1:, -self.nx//2+1:] = x[self.ny//2+1:, self.nx//2+1:]

        return xp

    def _chop(self, xp):
        '''
        This function chops the padded vorticity field containing the spurious
        wavenumbers resulting from the evaluation of the nonlinear term in the
        physical space.
        '''

        # --> Initialize variable.
        x = np.zeros((self.nx, self.ny)).astype('complex128')

        # --> Chop.
        # x[0:self.ny//2, 0:self.nx//2] = xp[0:self.ny//2, 0:self.nx//2]
        # x[0:self.ny//2, -self.nx//2:] = xp[0:self.ny//2, -self.nx//2:]
        # x[-self.ny//2:, 0:self.nx//2] = xp[-self.ny//2:, 0:self.nx//2]
        # x[-self.ny//2:, -self.nx//2:] = xp[-self.ny//2:, -self.nx//2:]

        x[0:self.ny//2, 0:self.nx//2] = xp[0:self.ny//2, 0:self.nx//2]
        x[0:self.ny//2, self.nx//2+1:] = xp[0:self.ny//2, -self.nx//2+1:]
        x[self.ny//2+1:, 0:self.nx//2] = xp[-self.ny//2+1:, 0:self.nx//2]
        x[self.ny//2+1:, self.nx//2+1:] = xp[-self.ny//2+1:, -self.nx//2+1:]

        return x

    def _compute_velocity(self, omega_hat):

        # --> Initialize various variables.
        uhat = np.zeros_like(omega_hat)         # x-velocity
        vhat = np.zeros_like(omega_hat)         # y-velocity

        psi_hat = np.zeros_like(omega_hat)      # Streamfunction.

        # --> Computation of the unpadded functions.
        for i in range(self.ny):
            for j in range(self.nx):
                # --> Solve for the stream function first. (No padding needed).
                if i == 0 and j == 0:
                    # NOTE: For alpha = beta = 0.
                    psi_hat[i, j] = 0.0
                else:
                    k2 = self.kx[j]**2 + self.ky[i]**2
                    psi_hat[i, j] = omega_hat[i, j] / k2

                # --> Unpadded velocity in spectral space.
                uhat[i, j] =  1j*self.ky[i]*psi_hat[i, j]
                vhat[i, j] = -1j*self.kx[j]*psi_hat[i, j]

        u = ifft2(uhat).real
        v = ifft2(vhat).real
        psi = ifft2(psi_hat).real

        return u, v, psi

if __name__ == '__main__':

    from scipy.io import savemat

    # --> Sets-up the simulation.
    simulation = NS2D(nx=256, ny=256, lx=1, ly=1, nu=5e-4, iostep=16, T=10, antialiasing=True)

    # --> Sets-up the initial condition.
    omega_hat = simulation.initial_condition(case=3, nmax=48, random_state=1234)
    # omega_hat = np.random.randn(simulation.nx, simulation.ny)
    # omega_hat -= np.mean(omega_hat)
    # omega_hat = fft2(omega_hat)
    # simulation.plot_figure(omega_hat, spectral=False, save=True)
    # output = {'vorticity': omega_hat}
    # savemat('initial_condition', output)

    fig = plt.figure(figsize=(4, 4), facecolor="black")
    camera = Camera(fig)

    # --> Run the simulation.
    for simulation.istep in trange(1, simulation.nsteps):
        omega_hat = simulation._rk4(omega_hat)
        if np.mod(simulation.istep, simulation.iostep) == 0:
            # --> Plots the current solution.
            simulation.plot_figure(fig, omega_hat)
            camera.snap()
            # # --> Save file.
            # output = {'vorticity': omega_hat}
            # savemat('output_%05i' %simulation.istep, output)

    anim = camera.animate()
    anim.save(
        "../imgs/2D_turbulence.mp4",
        fps=30, extra_args=["-vcodec", "libx264"]
        )

    plt.savefig("../imgs/2D_turbulence.png", bbox_inches="tight", dpi=300)

    plt.show()
