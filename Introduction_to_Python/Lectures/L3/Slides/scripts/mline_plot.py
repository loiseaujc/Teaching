import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 128)
y = np.cos(x)
z = np.sin(x)

plt.figure(facecolor="black")

plt.plot(x, y)
plt.plot(x, z)

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

plt.savefig("../imgs/mline_plot_default.png", bbox_inches="tight", dpi=1200)





plt.figure(facecolor="black")

plt.plot(x, y, label='cosinus')
plt.plot(x, z, label='sinus')

plt.legend()

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

plt.savefig("../imgs/mline_plot_legend.png", bbox_inches="tight", dpi=1200)





plt.figure(facecolor="black")

plt.plot(x, y, label='cosinus')
plt.plot(x, z, label='sinus', ls='--')

plt.legend()

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

plt.savefig("../imgs/mline_plot_style.png", bbox_inches="tight", dpi=1200)
