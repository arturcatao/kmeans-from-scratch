import numpy as np

class Kmeans:
    def __init__(self, k=None, max_iters=100, n_init=100, minK=1, maxK=10):
    # k = número de clusters, max_iters = limite de iterações para não rodar pra sempre
        self.k = k
        self.max_iters = max_iters
        self.n_init = n_init
        self.minK = minK
        self.maxK = maxK

    def fit(self, X):
        inertia = float("+inf")
        
        if self.k == None: self.k = self._find_best_k(X)
        
        for i in range(self.n_init):

            n_samples = X.shape[0]

            # escolhe k pontos aleatorios do dataset como centroides iniciais

            centroids = self._init_centroids_plusplus(X)

            for _ in range(self.max_iters):
                # atribuição: para cada ponto, encontra o centroide mais próximo
                clusters = np.zeros(n_samples, dtype=int)
                for j in range(n_samples):
                    distances = np.linalg.norm(X[j] - centroids, axis=1)
                    clusters[j] = np.argmin(distances)

                # guarda os centroides antes de atualizar para checar convergência
                prev_centroids = centroids.copy()

                # atualização: recalcula cada centroide como a media dos pontos do cluster
                for j in range(self.k):
                    points_values = X[clusters == j]
                    centroids[j] = np.mean(points_values, axis=0)
                #se os centroides não mudaram, o algoritmo convergiu
                if np.allclose(prev_centroids, centroids): break
            
            # escolhe a melhor combinação para o melhor modelo
            current_inertia = self._inertia(X, clusters, centroids) 
            if current_inertia < inertia:
                inertia = current_inertia
                self.centroids = centroids
                self.clusters = clusters
                self.inertia = inertia




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
    
    def _inertia(self, X, clusters, centroids):
        #faz com que o fit escolha a melhor combinação
        soma = 0
        for i in range(X.shape[0]):
            soma += np.sum((X[i] - centroids[clusters[i]]) ** 2)
        return soma
    
    def _find_best_k(self, X):
        inertias = []
        for k in range(self.minK, self.maxK + 1):
            model = Kmeans(k=k)
            model.fit(X)
            inertias.append(model.inertia)
        
        differences = np.diff(inertias)
        second_diff = np.diff(differences)
        best_k = self.minK + np.argmax(second_diff) + 2

        return best_k