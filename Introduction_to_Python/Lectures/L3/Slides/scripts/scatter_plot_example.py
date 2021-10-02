import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
x, y = iris.data, iris.target
labels = iris.target_names

fig, ax = plt.subplots(figsize=(4, 4))

for i in range(3):
    mask = y == i
    ax.scatter(x[mask, 0], x[mask, 1], color=plt.cm.tab20(i/3), ec="black")

ax.set(xlabel="Longueur des sépales", ylabel="Largeur des sépales")
ax.set(xticks=[], yticks=[])

ax.legend(
    # loc="center left",
    labels=labels.tolist(),
    facecolor="none",
    # labelcolor="white"
    # bbox_to_anchor=(1.02, 0.5)
    )

# ax.set_facecolor("black")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ax.spines["left"].set_color("white")
# ax.spines["bottom"].set_color("white")

# ax.xaxis.label.set_color("white")
# ax.yaxis.label.set_color("white")

# ax.tick_params(axis="both", colors="white")

plt.savefig("../imgs/iris_dataset.png", bbox_inches="tight", dpi=300)
plt.show()
