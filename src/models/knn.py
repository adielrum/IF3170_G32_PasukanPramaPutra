import numpy as np
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

class KNN(BaseModel):
    def __init__(self, k=3, metric='euclidean', voting_strategy='majority', *, p=3):

        self.k = k
        self.p = p

        if metric in ['euclidean', 'manhattan', 'minkowski', 'cosine']:
            self.metric = metric
        else:
            raise ValueError(f"Invalid metric: {metric}.")
        
        if voting_strategy in ['majority', 'distance']:
            self.voting_strategy = voting_strategy
        else:
            raise ValueError(f"Invalid voting strategy: {voting_strategy}.")
        
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):    
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes = np.unique(self.y_train)

        return self

    def predict(self, X):
        if self.X_train is None or self.y_train is None or self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        if self.X_train is None or self.y_train is None or self.classes is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        if len(X) == 0:
            return np.empty((0, len(self.classes)), dtype=float)
        
        probs = np.zeros((len(X), len(self.classes)))

        for i, x in enumerate(X):
            # Calculate distances
            distances = self._compute_distances(x)

            # Find k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]

            # Vote for class probabilities 
            scores = self._vote(k_labels, k_distances)
            probs[i] = scores / np.sum(scores)
        
        return probs

    def _compute_distances(self, x):
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet.")
        
        match self.metric:

            case 'euclidean':
                return np.linalg.norm(self.X_train - x, ord=2, axis=1)
            
            case 'manhattan':
                return np.linalg.norm(self.X_train - x, ord=1, axis=1)
            
            case 'minkowski':
                return np.linalg.norm(self.X_train - x, ord=self.p, axis=1)
            
            case 'cosine':
                dot = np.dot(self.X_train, x) 
                norm_train = np.linalg.norm(self.X_train, axis=1) 
                norm_x = np.linalg.norm(x)
                return 1.0 - (dot / (norm_train * norm_x + 1e-9))

            case _:
                raise ValueError(f"Invalid metric: {self.metric}.")
    
    def _vote(self, k_labels, k_distances):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        match self.voting_strategy:

            case 'majority':
                return np.bincount(k_labels, minlength=len(self.classes))
            
            case 'distance':
                weights = 1.0 / (k_distances + 1e-9)
                return np.bincount(k_labels, weights=weights, minlength=len(self.classes))

            case _:
                raise ValueError(f"Invalid voting strategy: {self.voting_strategy}.")