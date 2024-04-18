import numpy as np


class Simulation:
    def __init__(self, dk, STOP_K, n_landmarks, u = None):
        """
            Simulation class for encapsulating ground-truth position and landmarks

            Parameters
            ----------
            STOP_K : int
                Number of discrete time steps
            n_landmarks : int
                Number of landmarks
            u : np.array (optional)
                Control input [v, w].T
        """
        if u is None: self.u = np.array([1, 0.1])                       # [v, w].T
        else: self.u = u

        self._x = np.zeros((3, STOP_K))                   # [x, y, theta].T for each time step
        self._l_pos = Simulation.generate_landmarks(n_landmarks)

        self.x_dead_reckoning = np.zeros((3, STOP_K))      # [x, y, theta].T for each time step
        self.dk = dk

    def move(self, k):
        """
            Move the robot to the next position. 
            - Circular motion is assumed.

            Samples from the velocity motion model where v, w, theta are corrupted by Gaussian noise
            v' = v + gaussian(0, a_1 v^2 + a_2 w^2)
            w' = w + gaussian(0, a_3 v^2 + a_4 w^2)

            x, y are updated as center + r * [sin(theta + w'*dt), -cos(theta + w' * dt)].T
            a final rotation is added to theta

            theta' = theta + w' * dt + gaussian(0, a_5 v^2 + a_6 w^2) * dt
        """

        x_prev = self.x[:,k-1]
        u = self.u

        r_ = u[0] / u[1]
        x = x_prev[0] - r_ * np.sin(x_prev[2]) + r_ * np.sin(x_prev[2] + u[1]*self.dk)
        y = x_prev[1] + r_ * np.cos(x_prev[2]) - r_ * np.cos(x_prev[2] + u[1]*self.dk)
        theta = Simulation.angle(x_prev[2] + u[1]*self.dk)

        self.x[:,k] = np.array([[x, y, theta]])

    @property
    def x(self):
        return self._x

    @property
    def l_pos(self):
        return self._l_pos

    def observe(self):
        pass

    @staticmethod
    def angle(theta):
        return theta % (2*np.pi)

    @staticmethod
    def generate_landmarks(n_landmarks):
        l_x = np.random.uniform(-10, 10, size=(1, n_landmarks))             # [x, y].T for each landmark
        l_y = np.sqrt(100-l_x[:,:n_landmarks//2]**2) + 10
        l_y2 = -np.sqrt(100-l_x[:,n_landmarks//2:]**2) + 10
        l_y = np.hstack([l_y, l_y2])
        l_pos = np.vstack((l_x, l_y))
        l_pos += np.random.normal(-0.5, 1, size=l_pos.shape) + np.random.normal(0.5, 1, size=l_pos.shape)

        return l_pos

    