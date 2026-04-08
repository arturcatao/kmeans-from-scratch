import numpy as np

class Kmeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n_samples = X.shape[0]
    
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            clusters = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                distances = np.linalg.norm(X[i] - self.centroids, axis=1)
                clusters[i] = np.argmin(distances)
            self.clusters = clusters

            prev_centroids = self.centroids[:]
            for i in range(self.k):
                points_values = [X[clusters == i]]
                self.centroids[i] = np.mean(points_values, axis=0)
            
            if np.allclose(prev_centroids, self.centroids): break


    def predict(self, X):
        pass