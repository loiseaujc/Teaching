import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 128)
y = np.exp(x)

plt.figure(facecolor="black")

plt.plot(x, y)

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

plt.savefig("../imgs/scale_line_plot_default.png", bbox_inches="tight", dpi=1200)

plt.yscale('log')

plt.savefig("../imgs/scale_line_plot_logy.png", bbox_inches="tight", dpi=1200)
