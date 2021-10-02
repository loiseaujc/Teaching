import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

# Données génériques.
x = npr.randn(200, 2)

plt.figure(facecolor="black")
plt.scatter(x[:, 0], x[:, 1])

ax = plt.gca()
ax.spines["left"].set_color("white")
ax.spines["bottom"].set_color("white")
ax.spines["right"].set_color("white")
ax.spines["top"].set_color("white")


ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(axis="both", colors="white")
ax.set_facecolor("black")

plt.savefig("../imgs/scatter_plot_default.png", bbox_inches="tight", dpi=600)




plt.figure(facecolor="black")
plt.scatter(x[:, 0], x[:, 1], alpha=0.5)

ax = plt.gca()
ax.spines["left"].set_color("white")
ax.spines["bottom"].set_color("white")
ax.spines["right"].set_color("white")
ax.spines["top"].set_color("white")


ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(axis="both", colors="white")
ax.set_facecolor("black")

plt.savefig("../imgs/scatter_plot_alpha.png", bbox_inches="tight", dpi=600)



y = npr.randn(200, 2) + 1.0

plt.figure(facecolor="black")
plt.scatter(x[:, 0], x[:, 1], alpha=0.5)
plt.scatter(y[:, 0], y[:, 1], alpha=0.5)

ax = plt.gca()
ax.spines["left"].set_color("white")
ax.spines["bottom"].set_color("white")
ax.spines["right"].set_color("white")
ax.spines["top"].set_color("white")


ax.title.set_color("white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.tick_params(axis="both", colors="white")
ax.set_facecolor("black")

plt.savefig("../imgs/scatter_plot_multi.png", bbox_inches="tight", dpi=600)

plt.show()
