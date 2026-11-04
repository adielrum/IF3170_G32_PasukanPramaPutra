import numpy as np

class KNN:
    def __init__(self, k=3, metric='euclidean', voting_strategy='majority'):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.metric = None
        self.vote = None

        if metric == 'euclidean':
            self.metric = self.euclidean_distance
        elif metric == 'manhattan':
            self.metric = self.manhattan_distance
        elif metric == 'minkowski':
            self.metric = self.minkowski_distance
        elif metric == 'cosine':
            self.metric = self.cosine_similarity
        else:
            raise ValueError("Invalid distance metric")
        
        if voting_strategy == 'majority':
            self.vote = self.majority_vote
        elif voting_strategy == 'weighted':
            self.vote = self.weighted_vote
        else:
            raise ValueError("Invalid voting strategy")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        X = np.asarray(X)
        predictions = []

        # Fast vectorized path for euclidean distance
        if self.metric == self.euclidean_distance and isinstance(self.X_train, np.ndarray):
            for x in X:
                distances = np.linalg.norm(self.X_train - x, axis=1)
                k_indices = np.argpartition(distances, self.k)[:self.k]
                k_indices = k_indices[np.argsort(distances[k_indices])]
                k_nearest_labels = [(self.y_train[i], distances[i]) for i in k_indices]
                predicted_label = self.vote(k_nearest_labels)
                predictions.append(predicted_label)
            return np.array(predictions)

        # Fallback loop for arbitrary custom distance metrics
        for x in X:
            distances = [self.metric(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [(self.y_train[i], distances[i]) for i in k_indices]
            predicted_label = self.vote(k_nearest_labels)
            predictions.append(predicted_label)
        return np.array(predictions)

    # Voting Strategies

    # Majority Voting
    def majority_vote(self, k_nearest_labels):
        labels = [label for label, _ in k_nearest_labels]
        return max(set(labels), key=labels.count)

    # Weighted Voting
    def weighted_vote(self, k_nearest_labels):
        label_weights = {}
        for label, distance in k_nearest_labels:
            weight = 1 / (distance + 1e-5)  # Avoid division by zero
            if label in label_weights:
                label_weights[label] += weight
            else:
                label_weights[label] = weight
        return max(label_weights, key=label_weights.get)

    # Custom Distance Metrics

    # Custom distance function (Euclidean distance)
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Custom distance function (Manhattan distance)
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    # Custom distance function (Minkowski distance)
    def minkowski_distance(self, x1, x2, p=3):
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    # Custom distance function (Cosine similarity)
    def cosine_similarity(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2))  # Return distance as 1 - similarity

    # Custom distance function (Mahalanobis distance)
    def mahalanobis_distance(self, x1, x2, cov_matrix):
        diff = x1 - x2
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))

if __name__ == '__main__':
    model = KNN(k=5, metric='euclidean', voting_strategy='majority')
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1, 1])
    model.fit(X, y)
    X_test = np.array([[2, 2], [6, 7]])
    predictions = model.predict(X_test)
    print(predictions)  # Output: [0 1]
