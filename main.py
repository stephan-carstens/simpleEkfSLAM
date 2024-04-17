import numpy as np
import matplotlib.pyplot as plt

import EKF


# k == discrete time step number
STOP_K = 40
k = 0
dk = 1

n_landmarks = 22


def angle(theta):
    return theta % (2*np.pi)


def generate_landmarks(n_landmarks):
    l_x = np.random.uniform(-10, 10, size=(1, n_landmarks))             # [x, y].T for each landmark
    l_y = np.sqrt(100-l_x[:,:n_landmarks//2]**2) + 10
    l_y2 = -np.sqrt(100-l_x[:,n_landmarks//2:]**2) + 10
    l_y = np.hstack([l_y, l_y2])
    l_pos = np.vstack((l_x, l_y))
    l_pos += np.random.normal(-0.5, 1, size=l_pos.shape) + np.random.normal(0.5, 1, size=l_pos.shape)

    return l_pos

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


def move(x_prev, u):
    r_ = u[0] / u[1]
    x = x_prev[0] - r_ * np.sin(x_prev[2]) + r_ * np.sin(x_prev[2] + u[1]*dk)
    y = x_prev[1] + r_ * np.cos(x_prev[2]) - r_ * np.cos(x_prev[2] + u[1]*dk)
    theta = angle(x_prev[2] + u[1]*dk)

    return np.array([[x, y, theta]])


def main():
    x = np.zeros((3, STOP_K))                   # [x, y, theta].T for each time step
    u = np.array([1, 0.1])                       # [v, w].T

    l_pos = generate_landmarks(n_landmarks)

    for k in range(STOP_K):
        x[:,k] = move(x[:,k-1], u)

        update_plot(x, l_pos, k)


if __name__ == '__main__':
    main()