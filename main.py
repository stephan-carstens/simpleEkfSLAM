import simulation
from plotter import update_plot
# import EKF


def main():
    STOP_K = 60                                                     # k == discrete time step number

    velocity_motion_params = {
        'a_1': 0.01, 'a_2': 0.01, 
        'a_3': 0.01, 'a_4': 0.01, 
        'a_5': 0.01, 'a_6': 0.01}

    sim = simulation.Simulation(
        dk=1., STOP_K=STOP_K, n_landmarks=32, motion_params=velocity_motion_params)

    for k in range(STOP_K):
        sim.move(k)

        sim._observe(k)

        update_plot(sim.x, sim.x_dead_reckoning, sim.l_pos, sim.observed, k)
    plt.show()

if __name__ == '__main__':
    main()