import numpy as np
import matplotlib.pyplot as plt


def func(A, b):
    return lambda x : A @ x + b

def IFS(functions, probs, maxiter=100_000):

    # --> Define the IFS.
    func = lambda x : np.random.choice(functions, p=probs)(x)

    # --> Generate the IFS attractor.
    x = np.zeros((maxiter, 2))

    for i in range(1, maxiter):
        x[i] = func(x[i-1])

    return x

def plot_ifs(x, color="green"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.scatter(x[:, 0], x[:, 1], s=0.001, c=color)

    ax.set_aspect("equal")
    ax.axis(False)

def sierpinsky_triangle():
    # --> Sierpinsky triangle.
    h, w = np.sqrt(3)/2, 1

    A = 0.5 * np.eye(2)
    bs = np.array([0, 0]), np.array([w/2, 0]), np.array([w/4, h/2])

    maxiters = [1000, 10_000, 100_000]
    for i, maxiter in enumerate(maxiters):
        x = IFS([func(A, b) for b in bs], np.ones(3)/3, maxiter=maxiter)
        plot_ifs(x, color="black")
        plt.savefig("../imgs/sierpinsky_triangle_{0}.png".format(i+1), bbox_inches="tight", dpi=1200, transparent=True)
    plt.show()

def sierpinsky_carpet():
    # --> Sierpinsky carpet.
    A = 1/3 * np.eye(2)
    bs = [
        np.array([0, 0]), np.array([1/3, 0]),
        np.array([2/3, 0]), np.array([0, 1/3]),
        np.array([0, 2/3]), np.array([1/3, 2/3]),
        np.array([2/3, 1/3]), np.array([2/3, 2/3])
    ]

    maxiters = [10000, 100_000, 1_000_000]
    for i, maxiter in enumerate(maxiters):
        x = IFS([func(A, b) for b in bs], np.ones(8)/8, maxiter=maxiter)
        plot_ifs(x, color="black")
        plt.savefig("../imgs/sierpinsky_carpet_{0}.png".format(i+1), bbox_inches="tight", dpi=1200, transparent=True)
    plt.show()

def barnsley_fern():
    # --> Barnsley fern.
    A1, b1 = np.array([[0, 0], [0, 0.16]]), np.array([0, 0])
    A2, b2 = np.array([[0.85, 0.04], [-0.04, 0.85]]), np.array([0, 1.6])
    A3, b3 = np.array([[0.2, -0.26], [0.23, 0.22]]), np.array([0, 1.6])
    A4, b4 = np.array([[-0.15, 0.28], [0.26, 0.24]]), np.array([0, 0.44])

    f1 = lambda x : A1 @ x + b1
    f2 = lambda x : A2 @ x + b2
    f3 = lambda x : A3 @ x + b3
    f4 = lambda x : A4 @ x + b4

    probs = [0.01, 0.85, 0.07, 0.07]
    maxiters = [1000, 10_000, 100_000]
    for i, maxiter in enumerate(maxiters):
        x = IFS([f1, f2, f3, f4], probs, maxiter=maxiter)
        plot_ifs(x)
        plt.savefig("../imgs/barnsley_fern_{0}.png".format(i+1), bbox_inches="tight", dpi=1200, transparent=True)
    plt.show()

def fractal_tree():
    # --> List of linear transformations.
    def matrix(α, β, θ, φ):
        A = np.array([[α*np.cos(θ), -β*np.sin(φ)], [α*np.sin(θ), β*np.cos(φ)]])
        return A

    As = [
        matrix(0.05, 0.6, 0, 0),
        matrix(0.05, -0.5, 0, 0),
        matrix(0.6, 0.5, 0.698, 0.698),
        matrix(0.5, 0.45, 0.349, 0.3492),
        matrix(0.5, 0.55, -0.524, -0.524),
        matrix(0.55, 0.4, -0.698, -0.698),
    ]

    bs = [
        np.array([0, 0]), np.array([0, 1]), np.array([0, 0.6]),
        np.array([0, 1.1]), np.array([0, 1]), np.array([0, 0.7])
    ]

    maxiters = [1000, 10_000, 100_000]
    for i, maxiter in enumerate(maxiters):
        x = IFS([func(A, b) for A, b in zip(As, bs)], np.ones(6)/6, maxiter=maxiter)
        plot_ifs(x, color="black")
        plt.savefig("../imgs/fractal_tree_{0}.png".format(i+1), bbox_inches="tight", dpi=1200, transparent=True)
    plt.show()

if __name__ == "__main__":
    fractal_tree()
