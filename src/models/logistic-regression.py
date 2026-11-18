import numpy as np
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.01, num_iterations=1000, batch_size=None, *, l1_lambda=0.0, l2_lambda=0.0, random_state=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size

        self.weights = None
        self.bias = None

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.random_state = random_state

    def _softmax(self, z):
        # Stable softmax from https://jaykmody.com/blog/stable-softmax/ 
        c = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - c) 
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        
        y = self.predict_proba(X)
        return np.argmax(y, axis=1)

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")

        z = X @ self.weights + self.bias
        return self._softmax(z)

    def fit(self, X, y):
        # Main source: https://web.stanford.edu/~jurafsky/slp3/old_dec20/5.pdf 

        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        rng = np.random.default_rng(self.random_state)

        if self.batch_size is None:
            batch_size = 1
        elif self.batch_size >= num_samples:
            batch_size = num_samples
        elif self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")
        else:
            batch_size = self.batch_size

        # One-hot encoding
        y_reshaped = np.zeros((num_samples, num_classes))
        for i, label in enumerate(y):
            y_reshaped[i, label] = 1

        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

        for _ in range(self.num_iterations):
            indices = rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_reshaped[indices]

            y_predicted = self.predict_proba(X)

            # Mini-Batch Gradient Descent 
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                current_batch_size = len(X_batch)

                y_predicted = self.predict_proba(X_batch)

                # Gradients: https://dsplog.com/2025/06/22/gradients-for-multi-class-classification-with-softmax/#gsc.tab=0 
                dw = (1 / current_batch_size) * (X_batch.T @ (y_predicted - y_batch)) # 1/N * Xᵀ * (y - ŷ)
                db = (1 / current_batch_size) * np.sum(y_predicted - y_batch, axis=0) # 1/N * 1ᵀ * (y - ŷ)

                # Regularization : https://mbrenndoerfer.com/writing/elastic-net-regularization-complete-guide-mathematical-foundations-python-implementation
                dl1 = self.l1_lambda * np.sign(self.weights) # L1  
                dl2 = self.l2_lambda * self.weights # L2  

                self.weights -= self.learning_rate * (dw + dl1 + dl2) 
                self.bias -= self.learning_rate * db