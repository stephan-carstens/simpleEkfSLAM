import numpy as np
import matplotlib.pyplot as plt

import simulation
import EKF


# k == discrete time step number
STOP_K = 40
k = 0
dk = 1

n_landmarks = 22


def update_plot(x, l_pos, k):
    """
        Update the plot with the new state and trace of history up to time step k

        Parameters
        ----------
        x : np.array
            [x, y, theta].T for each time step (3, STOP_K)
        l_pos : np.array
            [x, y].T for each landmark (2, n_landmarks)
        k : int
            time step number
    """

    plt.cla()

    plt.plot(x[0, :k+1], x[1, :k+1], 'b-')          # history
    plt.plot(x[0, k], x[1, k], 'o', color='red')    # current position
    plt.plot(l_pos[0, :], l_pos[1, :], 'o', color='green')        # landmarks

    plt.pause(0.2)


def main():
    sim = simulation.Simulation(dk=dk, STOP_K=STOP_K, n_landmarks=n_landmarks)

    for k in range(STOP_K):
        sim.move(k)

        update_plot(sim.x, sim.l_pos, k)


if __name__ == '__main__':
    main()