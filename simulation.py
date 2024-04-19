import numpy as np


class Simulation:
    def __init__(self, dk:float, STOP_K:int, n_landmarks:int, 
                    motion_params:dict, observation_params:dict, u:np.array = None):
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
        self._l_pos = Simulation.generate_landmarks_uniform(n_landmarks)
        self.x_dead_reckoning = np.zeros((3, STOP_K))      # [x, y, theta].T for each time step
        self.dk = dk
        self.a = np.array([[motion_params['a_1'], motion_params['a_2']],
                           [motion_params['a_3'], motion_params['a_4']],
                           [motion_params['a_5'], motion_params['a_6']]])
        self.observation_params = observation_params
        self.observed = None

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
        self._move_dead_reckoning(k)
        self._move(k)
    
    def _move_dead_reckoning(self, k):
        x_prev = self.x_dead_reckoning[:,k-1]

        r_ = self.u[0] / self.u[1]
        x = x_prev[0] - r_*np.sin(x_prev[2]) + r_*np.sin(x_prev[2] + self.u[1]*self.dk)
        y = x_prev[1] + r_*np.cos(x_prev[2]) - r_*np.cos(x_prev[2] + self.u[1]*self.dk)
        theta = Simulation.angle(x_prev[2] + self.u[1]*self.dk)

        self.x_dead_reckoning[:,k] = np.array([[x, y, theta]])

    def _move(self, k):
        x_prev = self.x[:,k-1]

        v = self.u[0] + np.random.normal(0, np.sqrt(self.a[0,:].dot(self.u**2)))        # variance = a_1 v^2 + a_2 w^2
        w = self.u[1] + np.random.normal(0, np.sqrt(self.a[1,:].dot(self.u**2)))
        gamma =         np.random.normal(0, np.sqrt(self.a[2,:].dot(self.u**2)))

        r_ = v / w
        x = x_prev[0] - r_ * np.sin(x_prev[2]) + r_ * np.sin(x_prev[2] + w*self.dk)
        y = x_prev[1] + r_ * np.cos(x_prev[2]) - r_ * np.cos(x_prev[2] + w*self.dk)
        theta = Simulation.angle(x_prev[2] + w*self.dk + gamma*self.dk)

        self.x[:,k] = np.array([[x, y, theta]])

    def _observe(self, k):
        diff = self.x[:2,k][:,np.newaxis] - self._l_pos
        dists = np.hypot(diff[0], diff[1])
        observed = np.where(dists < 7)
        bearing = Simulation.angle(np.arctan2(diff[1, observed], diff[0, observed]) - self.x[2,k])
        
        return observed, bearing, dists

    def observe(self):
        observed, bearing, dists = self._observe(k)

        bearing +=  np.random.normal(0, self.observation_params["sigma_r"], size=bearing.shape)
        dists   +=  np.random.normal(0, self.observation_params["sigma_phi"], size=dists.shape)

        return observed, bearing, dists
        

    # Properties
    @property
    def x(self):
        return self._x

    @property
    def l_pos(self):
        return self._l_pos


    # Static methods
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

    @staticmethod
    def generate_landmarks_uniform(n_landmarks):
        l_pos = np.random.uniform(-20, 20, size=(2, n_landmarks))             # [x, y].T for each landmark

        return l_pos

    