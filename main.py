import numpy as np
import matplotlib.pyplot as plt

import simulation
import EKF


def update_plot(x_ground, x_dead_reckoning, l_pos, k):
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

    plt.plot(x_ground[0, :k+1], x_ground[1, :k+1], 'b-')                          # history
    plt.plot(x_dead_reckoning[0, :k+1], x_dead_reckoning[1, :k+1], color='black')                          # history
    plt.plot(x_ground[0, k], x_ground[1, k], 'o', color='red')                    # current position
    plt.plot(l_pos[0, :], l_pos[1, :], 'o', color='green')          # landmarks

    plt.pause(0.2)


def main():
    STOP_K = 40                                                     # k == discrete time step number

    velocity_motion_params = {
        'a_1': 0.01, 'a_2': 0.01, 
        'a_3': 0.01, 'a_4': 0.01, 
        'a_5': 0.01, 'a_6': 0.01}

    sim = simulation.Simulation(
        dk=1., STOP_K=STOP_K, n_landmarks=22, motion_params=velocity_motion_params)

    for k in range(STOP_K):
        sim.move(k)

        update_plot(sim.x, sim.x_dead_reckoning, sim.l_pos, k)


if __name__ == '__main__':
    main()