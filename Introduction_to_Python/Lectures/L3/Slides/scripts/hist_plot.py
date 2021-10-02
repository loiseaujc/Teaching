import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

# Données génériques.
x = npr.randn(1000)
y = npr.randn(100) + 1

plt.figure(facecolor="black")
plt.hist(x)

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

plt.savefig("../imgs/hist_plot_default.png", bbox_inches="tight", dpi=600)





plt.figure(facecolor="black")
plt.hist(x)
plt.hist(y)

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

plt.savefig("../imgs/hist_plot_multi.png", bbox_inches="tight", dpi=600)

plt.figure(facecolor="black")
plt.hist(x, alpha=0.5, density=True)
plt.hist(y, alpha=0.5, density=True)

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

plt.savefig("../imgs/hist_plot_density.png", bbox_inches="tight", dpi=600)
