import numpy as np

import matplotlib.pyplot as plt
FIG_WIDTH = 6
params = {'text.usetex': True,
          'font.size': 10,}
plt.rcParams['text.latex.preamble'] = [r'\usepackage{nicefrac}', r'\usepackage{bm}', r'\usepackage{upgreek}']
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams.update(params)

from scipy.optimize import root_scalar

@np.vectorize
def darcy_weisbach(Re, eps_D, f0=0.01):
    # --> Define the Colebrook equation.
    func = lambda x : 1/np.sqrt(x) + 2 * np.log10(eps_D/3.7 + 2.51/(Re*np.sqrt(x)))

    # --> Solve for the root.
    return root_scalar(func, bracket=[0.001, 0.1]).root

if __name__ == "__main__":

    Re = np.logspace(3, 8)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH/2, FIG_WIDTH))

    for i, eps_D in enumerate(np.logspace(-4, -1.5, 11)[::-1]):
        # --> Compute the Darcy-Weisbach friction factor for given eps_D
        f = darcy_weisbach(Re, eps_D)

        # -->
        ax.loglog(Re, f, label="{:.1g}".format(eps_D), color=plt.cm.cool_r(i/11))

    ax.set(xlabel=r"$Re$", xlim=(1e3, 1e8))

    ax.set(ylabel="$f_D$", ylim=(0.01, 0.1))

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title=r"$\nicefrac{\epsilon}{D}$")
    plt.savefig("../imgs/Moody_chart.png", bbox_inches="tight", dpi=300)
    plt.show()
