from .utils import xyah_to_tlwh

class Track:
    def __init__(self, mean, cov, track_id, n_init=3, max_age=30):
        self.mean = mean
        self.covariance = cov
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = "Tentative"
        self.features = []
        self.n_init = n_init
        self.max_age = max_age

    @property
    def is_confirmed(self):
        return self.state == "Confirmed"

    @property
    def is_deleted(self):
        return self.state == "Deleted"

    @property
    def tlwh(self):
        return xyah_to_tlwh(self.mean[:4])

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, det_tlwh, feature_vec):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, det_tlwh)
        if feature_vec is not None:
            self.features.append(feature_vec)
        self.hits += 1
        self.time_since_update = 0
        if self.state == "Tentative" and self.hits >= self.n_init:
            self.state = "Confirmed"

    def mark_missed(self):
        if self.state == "Tentative":
            self.state = "Deleted"
        elif self.time_since_update > self.max_age:
            self.state = "Deleted"
