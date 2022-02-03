import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

def generate_walker(width, height):
    return RandomWalker([np.random.randint(width), height])

def init_lattice_coral(w, h):

    # --> Initialize lattice to zero state.
    lattice = np.zeros((w, h), bool)

    # --> Initialize bottom layer.
    lattice[:, 0] = 1

    return lattice

class RandomWalker():

    def __init__(self, position, stick=False, stickprob=1):
        self.dim = len(position)
        self.position = position
        self.stick = stick
        self.stickprob = stickprob

    def step(self):
        # --> Possible direction walker can take.
        directions = [-1, 0, 1]

        # --> Update position.
        self.position += np.random.choice(directions, self.dim)

        return self

class DLA():
    def __init__(self, width=128, height=128, init_lattice=init_lattice_coral, periodicity="spanwise"):

        # --> Width and height of the lattice.
        self.width = width
        self.height = height

        # --> Initialize the lattice.
        self.lattice = init_lattice(width, height)

        # --> Periodic direction.
        self.periodicity = periodicity

    def simulate(self, nwalkers=100, genwalker=generate_walker):

        for i in trange(nwalkers):

            # --> Check lattice.
            if self._stop_dla():
                break

            # --> Generate new walker.
            walker = genwalker(self.width, self.height)

            # -->
            aggregate = False

            while aggregate is False:

                # --> Random Walk.
                walker.step()

                # --> Impose the periodicity.
                self._check_walker_pos(walker)

                # -->
                aggregate = self._check_aggregate(walker)

        return self

    def _stop_dla(self):
        if self.periodicity == "spanwise":
            return np.sum(self.lattice[:, self.height-1]) > 0

    def _check_walker_pos(self, walker):
        # --> Current position of the walker.
        x, y = walker.position

        # --> Impose spanwise periodicity.
        if self.periodicity == "spanwise":
            if x == self.width:
                walker.position[0] = 0

            if x == -1:
                walker.position[0] = self.width-1

                # --> Remove walker to far up to speed-up computation.
            if y == self.height:
                walker.position[1] = self.height-1

    def _check_aggregate(self, walker):

        # --> Get current position of the walker.
        idx, idy = walker.position

        # --> Check if it has neighbours in the lattice.
        if np.sum(self.lattice[idx-1:idx+2, idy-1:idy+2]) > 0:
            self.lattice[idx, idy] = 1
            return True
        else:
            return False


if __name__ == "__main__":

    # -->
    width, height = 512, 256

    # -->
    dla = DLA(width=width, height=height)
    dla.simulate(nwalkers=4096)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.matshow(dla.lattice.T[::-1], cmap=plt.cm.binary)

    ax.set_aspect("equal")
    # ax.axis(False)

    plt.show()
