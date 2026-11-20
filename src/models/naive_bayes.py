import numpy as np 

try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel


class NaiveBayes(BaseModel):
    def __init__(self, alpha=1.0, n_bins=3):
        self.alpha = alpha
        self.n_bins = n_bins

    @staticmethod
    def is_missing(value):
        return value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        )

    def prep_X(self, X, fitting=False):
        if hasattr(X, "columns"):
            if fitting:
                self.feature_names_ = list(X.columns)
            else:
                X = X[self.feature_names_]
            return X.to_numpy(dtype=object)

        X = np.asarray(X, dtype=object)
        if fitting:
            self.feature_names_ = list(range(X.shape[1]))
        return X

    def num_feat(self, X):
        numeric_features = set()

        for j, feature in enumerate(self.feature_names_):
            values = []
            is_numeric = True

            for value in X[:, j]:
                if self.is_missing(value):
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    is_numeric = False
                    break

            if is_numeric and len(set(values)) > self.n_bins:
                numeric_features.add(feature)

        return numeric_features

    def fit_bin_edges(self, X):
        self.bin_edges_ = {}
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]

        for j, feature in enumerate(self.feature_names_):
            if feature in self.numeric_features_:
                values = [
                    float(value) for value in X[:, j]
                    if not self.is_missing(value)
                ]
                self.bin_edges_[feature] = np.unique(
                    np.quantile(values, quantiles)
                )

    def transform(self, feature, value):
        if self.is_missing(value):
            return None

        if feature in self.numeric_features_:
            bin_index = np.searchsorted(
                self.bin_edges_[feature], float(value), side="left"
            )
            return f"bin_{bin_index}"

        return str(value).strip()

    def fit(self, X, y):
        X = self.prep_X(X, fitting=True)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.numeric_features_ = self.num_feat(X)
        self.fit_bin_edges(X)

        self.class_count_ = np.bincount(y_encoded, minlength=n_classes)
        self.class_prior_ = self.class_count_ / n_samples

        self.feature_count_ = [
            [dict() for _ in range(n_classes)]
            for _ in range(n_features)
        ]
        self.feature_values_ = [set() for _ in range(n_features)]

        for i in range(n_samples):
            class_index = y_encoded[i]

            for j, feature in enumerate(self.feature_names_):
                value = self.transform(feature, X[i, j])
                self.feature_values_[j].add(value)

                counts = self.feature_count_[j][class_index]
                counts[value] = counts.get(value, 0) + 1

        return self

    def peluang_bersyarat(self, feature_index, class_index, value):
        counts = self.feature_count_[feature_index][class_index]
        count = counts.get(value, 0)
        k = len(self.feature_values_[feature_index]) + 1

        return (count + self.alpha) / (
            self.class_count_[class_index] + self.alpha * k
        )

    def predict_proba(self, X):
        X = self.prep_X(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_scores = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for c in range(n_classes):
                score = np.log(self.class_prior_[c])

                for j, feature in enumerate(self.feature_names_):
                    value = self.transform(feature, X[i, j])
                    probability = self.peluang_bersyarat(j, c, value)
                    score += np.log(probability)

                log_scores[i, c] = score

        log_scores -= np.max(log_scores, axis=1, keepdims=True)
        probabilities = np.exp(log_scores)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
