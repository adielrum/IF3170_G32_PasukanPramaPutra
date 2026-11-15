# Implementasi model decision tree Classifier C4.5

import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Stopping criteria
        if num_samples == 0:
            return 0
        if len(unique_labels) == 0:
            return 0
        if len(unique_labels) == 1:
            return unique_labels[0]
        if num_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()  # Return the most common label

        # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, feature_column, y, threshold):
        parent_entropy = self._entropy(y)

        left_indices = feature_column < threshold
        right_indices = feature_column >= threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n_left = len(y[left_indices])
        n_right = len(y[right_indices])
        n_total = len(y)

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        gain = parent_entropy - (n_left / n_total) * left_entropy - (n_right / n_total) * right_entropy
        return gain

    def _entropy(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to avoid log(0)

    def _predict_sample(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree  # Leaf node

        feature, threshold, left_subtree, right_subtree = tree
        if sample[feature] < threshold:
            return self._predict_sample(sample, left_subtree)
        else:
            return self._predict_sample(sample, right_subtree)

    def print_tree(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, tuple):
            print("\t" * depth + f"Leaf: {tree}")
            return

        feature, threshold, left_subtree, right_subtree = tree
        print("\t" * depth + f"[Feature {feature} < {threshold}]")
        self.print_tree(left_subtree, depth + 1)
        self.print_tree(right_subtree, depth + 1)

if __name__ == '__main__':
    model = DecisionTree(max_depth=3)
    X = np.array([[2, 3, 4, 5], [1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]])
    y = np.array([0, 0, 1, 1, 1])
    model.fit(X, y)
    X_test = np.array([[3, 2, 1, 0], [5, 2, 4, 3]])
    predictions = model.predict(X_test)
    print(predictions)
    model.print_tree()
