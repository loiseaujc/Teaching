import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import root_scalar

@np.vectorize
def darcy_weisbach(Re, eps_D):
    func = lambda f : 1/np.sqrt(f) + 2 * np.log10(eps_D/3.7 + 2.51/(Re*np.sqrt(f)))
    return root_scalar(func, method="bisect", bracket=[1e-8, 0.1]).root

def moody_chart():

    Re = np.logspace(3, 8)
    eps_D = np.logspace(-4, -1.5, 9)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="black")

    for i, eps in enumerate(reversed(eps_D)):
        f = darcy_weisbach(Re, eps)
        ax.loglog(
            Re, f,
            label="{:.1g}".format(eps),
            color=plt.cm.spring(i/len(eps_D))
        )

    ax.legend(title="$\epsilon/D$", loc="center left", bbox_to_anchor=(1.02, 0.5), facecolor="none", labelcolor="white")
    ax.set(xlabel="$Re$", xlim=(1e3, 1e8))
    ax.set(ylabel="$f_D$", ylim=(1e-2, 0.1))
    ax.set_facecolor("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.tick_params(axis="both", colors="white")

    plt.savefig("../imgs/moody_chart.png", bbox_inches="tight", dpi=1200)

if __name__ =="__main__":
    moody_chart()
    plt.show()
