import numpy as np

class EKF:
    def __init__(self, motion_model_covariance, observation_model_covariance):
        self.R = motion_model_covariance # TODO review
        self.Q = observation_model_covariance

        self.mu = np.array([[0.], [0.], [0.]])                # should be a column vector
        self.sigma = np.zeros((3, 3))
        self.n_landmarks = 0
        self.landmark_indices = []

    def predict_update(self, u, dk, observed, range, bearing):
        self.predict(u, dk)
        self.update(observed, range, bearing)

    def predict(self, u, dk):
        r_ = u[0]/u[1]

        # update mean noise-free
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

    def update(self, observed, range, bearing):
        for i, j in enumerate(observed):
            if j not in self.landmark_indices:
                self.landmark_indices.append(j)
                self.n_landmarks += 1

                mu_j = np.array([[self.mu[0,0] + range[i]*np.cos(bearing[i] + self.mu[2,0])],
                                 [self.mu[1,0] + range[i]*np.sin(bearing[i] + self.mu[2,0])]])
                self.mu = np.vstack((self.mu, mu_j))
                self.sigma = np.hstack(   (np.vstack((self.sigma, np.zeros((2, self.sigma.shape[1])))), np.zeros((2+self.sigma.shape[0], 2)))   )
            else:
                mu_j = self.mu[3+2*self.landmark_indices.index(j):3+2*self.landmark_indices.index(j)+2,:]
                
            delta = mu_j - self.mu[:2,:]
            q = delta.T @ delta

            delta = delta.T[0]
            z = np.array([  [np.sqrt(q[0,0])],
                            [np.arctan2(delta[1], delta[0]) - self.mu[2,0]]  ])

            F = np.zeros((3+2, 3+2*self.n_landmarks))
            F[0,0] = 1; F[1,1] = 1; F[2,2] = 1
            F[3, 3+2*self.landmark_indices.index(j)] = 1; F[4, 3+2*self.landmark_indices.index(j)+1] = 1

            q = q[0,0]
            H = 1/q*np.array([  [-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                                [delta[1], -delta[0], -q, -delta[1], delta[0]]]) @ F
            # (2, 5) @ (5, 3+2*N_LANDMARKS) = (2, 3+2*N_LANDMARKS)

            # (3+2N, 3+2N) @ (2, 3+2N).T @ ((3, 3+2N) @ (3+2N,3+2N) @ (3, 3+2N).T + (3, 3))**-1
            K = self.sigma @ H.T @ np.linalg.inv(H @ self.sigma @ H.T + self.Q)
            
            # (3+2N, 1) = (3+2N, 1) + (3+2N, 3) @ (2, 1)
            self.mu += K @ (z - np.array([[range[i]], [bearing[i]]]))
            self.sigma = (np.eye(3 + 2*self.n_landmarks) - K @ H) @ self.sigma




    def F(self):
        return np.hstack(( np.eye(3,3), np.zeros((3, 2*self.n_landmarks)) ))
