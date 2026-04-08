import numpy as np

class Kmeans:
    def __init__(self, k, max_iters=100):
    # k = número de clusters, max_iters = limite de iterações para não rodar pra sempre
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n_samples = X.shape[0]
    
        # escolhe k pontos aleatorios do dataset como centroides iniciais
        self.centroids = self._init_centroids_plusplus(X)

        for _ in range(self.max_iters):
            # atribuição: para cada ponto, encontra o centroide mais próximo
            clusters = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                distances = np.linalg.norm(X[i] - self.centroids, axis=1)
                clusters[i] = np.argmin(distances)
            self.clusters = clusters

            # guarda os centroides antes de atualizar para checar convergência
            prev_centroids = self.centroids[:]

            # atualização: recalcula cada centroide como a media dos pontos do cluster
            for i in range(self.k):
                points_values = X[clusters == i]
                self.centroids[i] = np.mean(points_values, axis=0)
            #se os centroides não mudaram, o algoritmo convergiu
            if np.allclose(prev_centroids, self.centroids): break


    def predict(self, X):
        #atribui cada ponto ao centróide mais próximo e retorna os clusters
        n_samples = X.shape[0]

        clusters = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.centroids, axis=1)
            clusters[i] = np.argmin(distances)
        return clusters
    
    def _init_centroids_plusplus(self, X):
        random_idx = np.random.randint(0, X.shape[0])
        centroids = [X[random_idx]]

        for i in range(1, self.k):
            distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids], axis=0)
            probabilities = distances / distances.sum()
            new_idx = np.random.choice(len(X), p=probabilities)
            centroids.append(X[new_idx])
        return np.array(centroids)