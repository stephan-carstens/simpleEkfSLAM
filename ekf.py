import numpy as np
from simulation import Simulation

class EKF:
    def __init__(self, motion_model_covariance, observation_model_covariance):
        self.R = motion_model_covariance
        self.Q = observation_model_covariance

        self.mu = np.array([[0.], [0.], [0.]])                
        self.sigma = np.zeros((3, 3))
        self.n_landmarks = 0
        self.landmark_indices = []

    def predict_update(self, u, dk, observed, bearing, range):
        self.predict(u, dk)
        self.update(observed, bearing, range)

    def predict(self, u, dk):
        r_ = u[0]/u[1]

        # update mean noise-free, by assuming circular motion
        self.mu = self.mu + self.F().T @ \
            np.array([  [-r_*np.sin(self.mu[2,0]) + r_*np.sin(self.mu[2,0] + u[1]*dk)],
                        [r_*np.cos(self.mu[2,0]) - r_*np.cos(self.mu[2,0] + u[1]*dk)],
                        [u[1]*dk]  ])
        # (3, 1) = (3, 1) + (3, 3+2*N_LANDMARKS).T @ (3,1)

        # update covariance
        g = np.array([  [0, 0, -r_*np.cos(self.mu[2,0]) + r_*np.cos(self.mu[2,0] + u[1]*dk)],
                        [0, 0, -r_*np.sin(self.mu[2,0]) + r_*np.sin(self.mu[2,0] + u[1]*dk)],
                        [0, 0, 0]  ])
        G = np.eye(3 + 2*self.n_landmarks) + self.F().T @ g @ self.F()
        # (3+2N, 3+2N) + (3, 3+2*N_LANDMARKS).T @ (3, 3) @ (3, 3+2*N_LANDMARKS)

        self.sigma += G.T @ self.sigma @ G + self.F().T @ self.R @ self.F() 

    def update(self, observed, bearing, range):
        for l_i, l_id in enumerate(observed):       # consistant l_id used for correspondence
            if l_id not in self.landmark_indices:
                self.landmark_indices.append(l_id)
                self.n_landmarks += 1

                mu_j = np.array([[self.mu[0,0] + range[l_i]*np.cos(Simulation.angle(bearing[l_i] + self.mu[2,0]))],
                                 [self.mu[1,0] + range[l_i]*np.sin(Simulation.angle(bearing[l_i] + self.mu[2,0]))]])

                # augment mu and sigma 
                self.mu = np.vstack((self.mu, mu_j))
                self.sigma = np.hstack(   (np.vstack((self.sigma, np.zeros((2, self.sigma.shape[1])))), np.zeros((2+self.sigma.shape[0], 2)))   )
                self.sigma[-2:, -2:] = np.eye(2) * 1e-10
            else:
                mu_j = self.mu[3+2*self.landmark_indices.index(l_id):3+2*self.landmark_indices.index(l_id)+2,:]

            delta = mu_j - self.mu[:2,:]
            q = delta.T @ delta
            q = q[0,0]

            z_hat = np.array([  [np.sqrt(q)],
                                [Simulation.angle(np.arctan2(delta[1, 0], delta[0, 0]) - self.mu[2,0])]  ])

            F = np.zeros((3+2, 3+2*self.n_landmarks))
            F[:3,:3] = np.eye(3)                                    # F[0,0] = 1; F[1,1] = 1; F[2,2] = 1
            F[3, 3+2*self.landmark_indices.index(l_id)] = 1 
            F[4, 3+2*self.landmark_indices.index(l_id)+1] = 1

            h = (1/q)*np.array([    [-np.sqrt(q)*delta[0, 0],   -np.sqrt(q)*delta[1, 0], 0, np.sqrt(q)*delta[0, 0], np.sqrt(q)*delta[1, 0]],
                                    [delta[1, 0],               -delta[0, 0],           -q, -delta[1, 0],           delta[0, 0]]])
            H = h @ F
            # (2, 5) @ (5, 3+2*N_LANDMARKS) = (2, 3+2*N_LANDMARKS)

            # (3+2N, 3+2N) @ (2, 3+2N).T @ ((2, 3+2N) @ (3+2N,3+2N) @ (2, 3+2N).T + (2, 2))**-1
            K = self.sigma @ H.T @ np.linalg.inv(H @ self.sigma @ H.T + self.Q)
            innovation = np.array([     [range[l_i] - z_hat[0,0]], 
                                        [Simulation.angle(bearing[l_i] - z_hat[1,0])] ]) 
            
            # (3+2N, 1) = (3+2N, 1) + (3+2N, 2) @ (2, 1)
            self.mu = self.mu + K @ innovation
            # self.mu += K @ (np.array([[range[l_i]], [bearing[l_i]]]) - z_hat)
            self.sigma = (np.eye(3 + 2*self.n_landmarks) - K @ H) @ self.sigma


    def F(self):
        return np.hstack(( np.eye(3,3), np.zeros((3, 2*self.n_landmarks)) ))
