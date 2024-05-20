import matplotlib.pyplot as plt

def update_plot(x_ground, x_dead_reckoning, est_x, l_pos, observed, k):
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
    ax = plt.gca()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 35)

    plt.plot(x_ground[0, :k+1], x_ground[1, :k+1], 'b-', label="true path", color="red")                                # history
    plt.plot(x_dead_reckoning[0, :k+1], x_dead_reckoning[1, :k+1], color='black', label="command path")       # history
    plt.plot(x_ground[0, k], x_ground[1, k], 'o', color='red')                          # current position
    plt.plot(est_x[0], est_x[1], 'o', color='blue', label="EKF")                          # current position

    plt.plot(l_pos[0, :], l_pos[1, :], 'o', color='gray')                              # landmarks
    plt.plot(l_pos[0, observed], l_pos[1, observed], 'o', color='green', alpha=0.5)                              # landmarks

    plt.legend()
    plt.pause(0.2)