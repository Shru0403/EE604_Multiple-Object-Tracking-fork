import numpy as np

class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)
        std = [
            2 * self._std_weight_pos * measurement[3],
            2 * self._std_weight_pos * measurement[3],
            1e-1,
            2 * self._std_weight_pos * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            1e-3,
            10 * self._std_weight_vel * measurement[3],
        ]
        covariance = np.diag(np.square(np.array(std, dtype=np.float32)))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3],
        ]
        std_vel = [
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]).astype(np.float32))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean, covariance):
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        innovation_cov = np.diag(np.square(np.array(std, dtype=np.float32)))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(chol.T, np.linalg.solve(chol, (covariance @ self._update_mat.T).T)).T
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ self._update_mat @ covariance
        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]
        chol = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = np.linalg.solve(chol, d.T)
        return np.sum(z * z, axis=0)
