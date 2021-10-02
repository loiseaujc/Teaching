import os.path

import numpy as np

from scipy.linalg import norm
from scipy.optimize import minimize
from scipy.special import softmax as _softmax

def softmax(x):
    x = np.insert(x, x.shape[1], 0, axis=1)
    x -= np.max(x)
    return _softmax(x, axis=1)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from celluloid import Camera

import pickle

import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

from tqdm import trange

class SoftMax(BaseEstimator, ClassifierMixin):

    def __init__(self, solver="L-BFGS-B", l2=0.0, verbose=True, tol=1e-6, hist=True):
        self.l2 = l2
        self.solver = solver
        self.verbose = verbose
        self.tol = tol
        self.hist = hist

    def fit(self, X, y):

        # --> Reshape data if needed.
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        # --> Include constant term.
        X = np.insert(X, 0, 1, axis=1)

        # --> Number of features.
        n = X.shape[1]

        # --> One-Hot Encoding of the label vector.
        enc = OneHotEncoder(categories="auto", sparse=False)
        y = enc.fit_transform(y.reshape((-1, 1)))

        # --> Initialize weights.
        num_class = y.shape[1] - 1
        W = np.zeros((n, num_class))
        # W = np.random.randn(n, num_class)

        callback = None
        if self.hist:
            Ws = [W]
            callback = lambda x : Ws.append(x.reshape((n, num_class)))

        # --> Train the model.
        output = minimize(
            self.loss,
            W,
            jac = True,
            args = (X, y, self.l2),
            tol = self.tol,
            method = self.solver,
            callback=callback
        )

        # --> Print output of minimize.
        if self.verbose is True or output["success"] is False:
            print("----------------------------------------------")
            print("-----     Output of SciPy's minimize     -----")
            print("----------------------------------------------\n")
            print( output )

        # --> Save the model coefficients.
        W = output.x.reshape((n, num_class))
        self.b, self.W = W[0], W[1:]
        self.Ws = Ws

        return self

    def loss(self, W, X, y, l2):

        # --> Number of features and training examples.
        m, n = X.shape

        # --> Number of classses.
        num_class = y.shape[1] - 1

        # --> Reshape W for simplicity.
        W = W.reshape((n, num_class))

        # --> Evaluate model.
        y_pred = softmax(X @ W)

        # --> Compute loss function.
        ϵ = 1e-10
        loss = -np.sum(y * np.log(y_pred + ϵ)) / m

        # --> Add Thikonov regularization.
        loss += 0.5 * l2 * norm(W[1:, :])**2

        # --> Compute gradient of loss function.
        grad = X.T @ (y_pred[:, :-1] - y[:, :-1])
        grad /= m

        # --> Add Thikonov regularization.
        grad[1:, :] += l2*W[1:, :]

        return loss, grad.flatten()

    def decision_function(self, X):
        return X @ self.W + self.b

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

def plot_mnist(x_train):

    fig, axes = plt.subplots(
        3, 6,
        figsize=(10, 5),
        sharex=True, sharey=True,
        facecolor="black"
        )

    # --> Plot the first few digits.
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_train[i], cmap=plt.cm.gray)
        ax.axis("off")

    plt.savefig("../imgs/mnist_examples.png", bbox_inches="tight", dpi=600)

def make_plot(fig, axes, x, ypred):

    # --> Plot the digit.
    axes[0].imshow(x, cmap=plt.cm.binary_r)
    axes[0].axis("off")

    # -->  Plot the probability.
    axes[1].barh(
        np.arange(0, 10), ypred,
        color="lightgray"
    )

    axes[1].set_xticks([0, 0.5, 1])
    axes[1].set_xlabel("Probabilité", color="white")
    axes[1].set_yticks(np.arange(0, 10))
    axes[1].set_xlim(0, 1)

    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)

    axes[1].spines["left"].set_color("white")
    axes[1].spines["bottom"].set_color("white")

    axes[1].xaxis.label.set_color("white")
    axes[1].yaxis.label.set_color("white")

    axes[1].tick_params(axis="both", colors="white")

    axes[1].set_facecolor("black")

    plt.tight_layout()

def main():
    # --> Load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # --> Plot some examples.
    plot_mnist(x_train)

    # --> Reshape each 28x28 img into a 784 vector.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # --> Preprocessing : data scaling.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # --> Train the classifier.
    fname = "softmax_model.p"
    if os.path.isfile(fname):
        clf = pickle.load(open(fname, "rb"))
    else:
        clf = SoftMax(solver="CG", l2=0.125).fit(x_train, y_train)
        pickle.dump(clf, open(fname, 'wb'))

    # --> Create figure.
    fig = plt.figure(facecolor="black", figsize=(4, 4))
    camera = Camera(fig)

    # --> Set subplot grid.
    gs = GridSpec(2, 3)
    axes = [fig.add_subplot(gs[:, :-1]), fig.add_subplot(gs[:, -1])]
    invproc = lambda x : scaler.inverse_transform(x).reshape((28, 28))
    predict_proba = lambda x, W : softmax(x @ W[1:] + W[0])
    print(clf.Ws[0].shape)
    idx = 9
    for i in trange(len(clf.Ws)//2):
        ypred = predict_proba(x_test, clf.Ws[i])
        make_plot(fig, axes, invproc(x_test[idx]), ypred[idx])
        camera.snap()

    anim = camera.animate()
    anim.save("../imgs/mnist_one_digit.mp4")

    plt.savefig("../imgs/mnist_one_digit.png", bbox_inches="tight", dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
