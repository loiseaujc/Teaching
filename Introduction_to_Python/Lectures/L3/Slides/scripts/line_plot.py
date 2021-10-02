import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 128)
y = np.cos(x)

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

plt.savefig("../imgs/line_plot_default.png", bbox_inches="tight", dpi=1200)

plt.title('y = cos(x)')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig("../imgs/line_plot_labels.png", bbox_inches="tight", dpi=1200)

plt.xlim(-6, 6)
plt.ylim(-1, 1)

plt.savefig("../imgs/line_plot_limits.png", bbox_inches="tight", dpi=1200)





plt.figure(facecolor="black")

plt.plot(x, y, c='r', lw=3)

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

plt.title('y = cos(x)')
plt.xlabel('x')
plt.ylabel('y')

plt.xlim(-6, 6)
plt.ylim(-1, 1)

plt.savefig("../imgs/line_plot_color.png", bbox_inches="tight", dpi=1200)

plt.show()
