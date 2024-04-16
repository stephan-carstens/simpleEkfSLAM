import numpy as np
import matplotlib.pyplot as plt

import EKF


# k == discrete time step number
STOP_K = 40
k = 0


def update_plot(x, k):
    """
    Update the plot with the new state and trace of history up to time step k

    Parameters
    ----------
    x : np.array
        [x, y, theta].T for each time step (3, STOP_K)
    """
    plt.cla()

    plt.plot(x[0, :k+1], x[1, :k+1], 'b-')
    plt.plot(x[0, k], x[1, k], 'o', color='red')

    # plt.show()
    plt.pause(0.1)


def move(x_k):
    return x_k + np.random.uniform(-0.1, 0.1, 3)


def main():
    x = np.zeros((3, STOP_K))                   # [x, y, theta].T for each time step

    for k in range(STOP_K):
        x_k = x[:,k]

        x_k = move(x_k)

        update_plot(x, k)

        if k+1 < STOP_K: 
            x[:,k+1] = x_k


if __name__ == '__main__':
    main()