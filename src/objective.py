import numpy as np
import numpy.linalg as LA


class Objective:
    def __init__(self, x0, theta0, L) -> None:
        self.x0 = x0
        self.theta0 = theta0
        self.L = L
        if isinstance(x0, np.ndarray):
            self.x0 = self.x0.astype(float)
            self.theta0 = self.theta0.astype(float)
    
    def __call__(self, x, theta):
        if isinstance(x, np.ndarray):
            return LA.norm(x - self.x0, 1, axis=-1) + (self.L / (1 + np.exp(x @ theta.T)))
        else:
            return np.abs(x - self.x0) + (self.L / (1 + np.exp(x * theta)))
    
    def x1d(self, theta, c):
        x = np.log(((self.L*theta-2) + np.sqrt((self.L*theta-2)**2 - 4))/(2*c)) / theta
        return x
    
    def xnd(self, theta):
        x0_ = self.x0.squeeze()
        theta_ = theta.squeeze()
        idx = np.argmax(theta)
        c = np.sum([np.exp(x0_[i] * theta_[i]) for i in range(len(theta_)) if i != idx])
        x_ = self.x1d(theta_[idx], c)
        x = x0_.copy()
        x[idx] = x_
        x = x[np.newaxis]
        if np.all(theta_ == theta_[0]):
            mu = np.mean(x)
            x = np.full(x.shape, mu)
        return x if self(x, theta) < self(self.x0, theta) else self.x0
    
    def X(self, theta):
        if isinstance(theta, np.ndarray):
            return self.xnd(theta)
        else:
            return self.x1d(theta, 1)