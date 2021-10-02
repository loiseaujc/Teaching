import numpy as np

from tiobeindexpy import tiobeindexpy as tbpy

import matplotlib.pyplot as plt

# --> Get data.
data = tbpy.top_20()

# --> Clean-up data.
data["Ratings"] = data.loc[:, "Ratings"].apply(lambda x: float(x.strip("%")))
data["Programming Language"] = data["Programming Language.1"]

fig, ax = plt.subplots(1, 1, figsize=(8, 4), facecolor="black")

data.plot(
    y="Ratings", x="Programming Language",
    kind="barh",
    ax=ax,
    legend=False,
    fontsize=7,
    color=plt.cm.inferno_r(np.linspace(0, 1, 20))
    )

ax.invert_yaxis()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("white")
ax.spines["bottom"].set_color("white")

ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

ax.set_xlabel("Ratings (%)", fontsize=8)
ax.set_ylabel("")

ax.set_xlim(0, 13)

ax.tick_params(axis="both", colors="white")

ax.set_facecolor("black")

plt.savefig("../imgs/tiobe_ratings.png", bbox_inches="tight", dpi=1200)

plt.show()
