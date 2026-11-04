# Implementasi Logistic Regression Classifier
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        y_reshaped = np.zeros((num_samples, num_classes))
        for idx, label in enumerate(y):
            y_reshaped[idx, label] = 1
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._softmax(linear_model)
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y_reshaped))
            db = (1 / num_samples) * np.sum(y_predicted - y_reshaped, axis=0)
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._softmax(linear_model)
        return np.argmax(y_predicted, axis=1)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

if __name__ == '__main__':
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 2])
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)
    print("Train acc:", model.evaluate(X, y))
    X_test = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    print(model.predict(X_test))

