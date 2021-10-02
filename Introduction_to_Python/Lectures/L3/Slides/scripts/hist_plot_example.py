import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy import stats

iris = load_iris()
x, y = iris.data, iris.target
labels = iris.target_names

fig, ax = plt.subplots(figsize=(8, 2))

for i in range(3):
    mask = y == i
    _, bins, _ = ax.hist(x[mask, 0],
                         density=True,
                         color=plt.cm.tab20(i/3),
                         alpha=0.5)

    μ, σ = stats.norm.fit(x[mask, 0])
    z = np.linspace(bins.min()/2, 1.5*bins.max(), 256)
    best_fit = stats.norm.pdf(z, μ, σ)

    ax.plot(z, best_fit, color=plt.cm.tab20(i/3))

ax.set(xlabel="Longueur des sépales", ylabel="Densité de probabilité")
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

plt.savefig("../imgs/iris_dataset_bis.png", bbox_inches="tight", dpi=300)
plt.show()
