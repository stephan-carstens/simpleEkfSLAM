import simulation
from plotter import update_plot
import matplotlib.pyplot as plt
import ekf

import numpy as np


def main():
    STOP_K = 60                                                     # k == discrete time step number

    velocity_motion_params = {
        'a_1': 0.001, 'a_2': 0.001, 
        'a_3': 0.001, 'a_4': 0.001, 
        'a_5': 0.001, 'a_6': 0.001}
    observation_params = {
        'sigma_r': 0.1, 'sigma_phi': 0.01}
    sim = simulation.Simulation(
        dk=1., STOP_K=STOP_K, n_landmarks=100, 
        motion_params=velocity_motion_params, observation_params=observation_params)

    R = np.eye(3, 3) * 0.1
    Q = np.eye (2, 2) * 0.1
    est = ekf.EKF(motion_model_covariance=R, observation_model_covariance=Q)

    for k in range(STOP_K):
        sim.move(k)

        obs_ids, bearing, dist = sim.observe(k)
        est.predict_update(sim.u, sim.dk, obs_ids, bearing, dist)

        # obs_ids, _, _ = sim._observe(k)
        update_plot(sim.x, sim.x_dead_reckoning, est.mu[:2, 0], sim.l_pos, obs_ids, k)
    plt.show()

if __name__ == '__main__':
    main()