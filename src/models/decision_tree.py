import numpy as np
import numbers
import os
import matplotlib.pyplot as plt
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

class Node:
    def __init__(
        self,
        feature=None,
        children=None,
        threshold=None,
        children_probs=None,
        *,
        value=None,
        probs=None,
    ):
        self.feature = feature
        self.children = children 
        self.children_probs = children_probs 
        self.threshold = threshold
        self.value = value
        self.probs = probs

    @property
    def is_leaf(self):
        return self.value is not None

class DecisionTree(BaseModel):
    def __init__(
        self,
        criterion="entropy",
        use_gain_ratio=True,
        max_depth=100,
        min_samples_split=2,
        categorical_features=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Invalid criterion: {criterion}")

        self.criterion = criterion
        self.use_gain_ratio = bool(use_gain_ratio)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorical_features = categorical_features

        self.root = None
        self.classes = None
        self.categories = {}
    
    @staticmethod
    def _is_null_value(x):
        return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))

    @staticmethod
    def _is_num_value(x):
        return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))

    @classmethod
    def _is_null(cls, arr):
        return np.array([cls._is_null_value(x) for x in arr], dtype=bool)

    def _infer_feature_types(self, X):
        num_features = X.shape[1]
        categories = {}

        for j in range(num_features):
            col = X[:, j]
            valid_rows = col[~self._is_null(col)]

            if self.categorical_features is not None and j in self.categorical_features:
                is_cat = True
            elif len(valid_rows) == 0:
                is_cat = True
            else:
                is_cat = not all(self._is_num_value(v) for v in valid_rows)

            if is_cat:
                try:
                    categories[j] = np.unique(valid_rows)
                except TypeError: 
                    categories[j] = np.array(list(dict.fromkeys(valid_rows)), dtype=object) # in case of mixed dtypes

        return categories

    def _create_leaf(self, y_indices, weights):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")

        N = len(self.classes)
        weighted_counts = np.bincount(y_indices, weights=weights, minlength=N)
        total_counts = np.sum(weighted_counts)

        probs = weighted_counts / total_counts 
        majority = self.classes[np.argmax(weighted_counts)]

        return Node(value=majority, probs=probs)

    def _build_tree(self, X, y_indices, weights, depth=0):
        total_weight = np.sum(weights)
        unique_labels = np.unique(y_indices[weights > 0])

        # Stopping criteria
        if len(unique_labels) <= 1:
            return self._create_leaf(y_indices, weights)
        if total_weight < self.min_samples_split or depth >= self.max_depth:
            return self._create_leaf(y_indices, weights)

        # Select best split
        best_feature, best_threshold = self._best_split(X, y_indices, weights)

        if best_feature is None:
            return self._create_leaf(y_indices, weights)

        # Separate known and unknown values 
        null_mask = self._is_null(X[:, best_feature])
        known_mask = ~null_mask
        known_weight = np.sum(weights[known_mask])

        is_categorical = best_feature in self.categories

        # Categorical feature branching
        if is_categorical:
            child_masks = {
                val: known_mask & (X[:, best_feature] == val) 
                for val in self.categories[best_feature]
            }

        # Continuous feature branching
        else:
            col_known = np.zeros(len(X), dtype=float)
            col_known[known_mask] = X[known_mask, best_feature].astype(float)
            child_masks = {
                "left": known_mask & (col_known < best_threshold),
                "right": known_mask & (col_known >= best_threshold),
            }

        # Calculate weights 
        children = {}
        children_probs = {}

        for child_key, child_mask in child_masks.items():
            p_child = np.sum(weights[child_mask]) / known_weight
            children_probs[child_key] = p_child

            if p_child <= 0:
                children[child_key] = self._create_leaf(y_indices, weights)
            else:
                active_mask = child_mask | null_mask 
                child_weights = weights[active_mask].copy() # parent row weights  
                child_nulls = null_mask[active_mask] 
                child_weights[child_nulls] *= p_child # updated row weights  

                children[child_key] = self._build_tree(
                    X[active_mask],
                    y_indices[active_mask],
                    child_weights,
                    depth + 1,
                )

        return Node(
            feature=best_feature,
            children=children,
            children_probs=children_probs,
            threshold=best_threshold,
        )

    def _impurity_from_probs(self, probs):
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0

        match self.criterion:
            case "entropy":
                return -np.sum(probs * np.log2(probs))
            case "gini":
                return 1.0 - np.sum(probs ** 2)
            case _:
                raise ValueError(f"Invalid criterion: {self.criterion}.")

    def _impurity(self, y_indices, weights):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        total_weight = np.sum(weights)
        counts = np.bincount(y_indices, weights=weights, minlength=len(self.classes))
        probs = counts[counts > 0] / total_weight

        return self._impurity_from_probs(probs)

    def _gain(self, y_known, w_known, child_masks, w_unknown):
        w_known_total = np.sum(w_known)
        w_total = w_known_total + w_unknown

        # L(S)
        parent_impurity = self._impurity(y_known, w_known) 

        # Σᵥ |Sᵥ|/|S| L(Sᵥ) 
        child_weights = np.array([np.sum(w_known[mask]) for mask in child_masks], dtype=float)
        child_impurity = 0.0 
        for mask, w_child in zip(child_masks, child_weights):
            if w_child > 0:
                child_impurity += (w_child / w_known_total) * self._impurity(y_known[mask], w_known[mask])

        # L(S) + Σᵥ |Sᵥ|/|S| L(Sᵥ) 
        gain = (w_known_total / w_total) * (parent_impurity - child_impurity)
        if gain <= 0:
            return 0.0

        # Gain 
        if not self.use_gain_ratio:
            return gain

        # Split Impurity 
        partition_weights = np.append(child_weights, w_unknown) if w_unknown > 0 else child_weights
        split_impurity = self._impurity_from_probs(partition_weights / w_total)

        if split_impurity <= 1e-12:
            return 0.0

        # Gain Ratio 
        return gain / split_impurity

    def _best_continuous_split(self, col_known, y_known, w_known, w_unknown):
        unique_vals = np.unique(col_known[w_known > 0])
        if len(unique_vals) < 2:
            return 0.0, None

        thresholds = 0.5 * (unique_vals[:-1] + unique_vals[1:])
        best_score = 0.0
        best_thresh = None

        for t in thresholds:

            left_mask = col_known < t
            right_mask = ~left_mask
            score = self._gain(y_known, w_known, [left_mask, right_mask], w_unknown)

            if score > best_score:
                best_score = score
                best_thresh = t

        return best_score, best_thresh

    def _best_split(self, X, y_indices, weights):
        num_features = X.shape[1]
        best_score = 0.0
        best_feature = None
        best_threshold = None

        for j in range(num_features):
            col = X[:, j]
            null_mask = self._is_null(col)
            known_mask = ~null_mask

            w_known = weights[known_mask]
            if np.sum(w_known) <= 0:
                continue

            y_known = y_indices[known_mask]
            w_unknown = np.sum(weights[null_mask])

            # Categorical feature
            if j in self.categories:

                col_known = col[known_mask]
                child_masks = [col_known == val for val in self.categories[j]]
                score = self._gain(y_known, w_known, child_masks, w_unknown)
                threshold = None

            # Continuous feature
            else:
                col_known = col[known_mask].astype(float)
                score, threshold = self._best_continuous_split(col_known, y_known, w_known, w_unknown)

            if score > best_score:
                best_score = score
                best_feature = j
                best_threshold = threshold

        return best_feature, best_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        self.classes, y_indices = np.unique(y, return_inverse=True)
        self.categories = self._infer_feature_types(X)

        initial_weights = np.ones(len(y_indices), dtype=float)
        self.root = self._build_tree(X, y_indices, initial_weights, depth=0)

        return self

    def predict_proba(self, X):
        if self.root is None or self.classes is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=object)
        if len(X) == 0:
            return np.empty((0, len(self.classes)), dtype=float)

        return np.array([self._predict_sample_proba(sample, self.root) for sample in X], dtype=float)

    def predict(self, X):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    def _marginal_child_proba(self, sample, node):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        total_probs = np.zeros(len(self.classes), dtype=float)
        for child_key, child_node in node.children.items():
            p_child = node.children_probs[child_key]
            if p_child > 0:
                total_probs += p_child * self._predict_sample_proba(sample, child_node)
        return total_probs 

    def _predict_sample_proba(self, sample, node):
        if node.is_leaf:
            return node.probs

        val = sample[node.feature]

        # Missing value 
        if self._is_null_value(val):
            return self._marginal_child_proba(sample, node) 

        # Continuous split 
        if node.threshold is not None:
            child_key = "left" if float(val) < node.threshold else "right"
            return self._predict_sample_proba(sample, node.children[child_key])

        # Categorical split 
        if val in node.children:
            return self._predict_sample_proba(sample, node.children[val])

        # Fallback
        return self._marginal_child_proba(sample, node)

    def __str__(self):
        return self._print_tree(self.root)

    def _print_tree(self, node, depth=0):
        if node.is_leaf:
            return f"{'|   ' * depth}Leaf: {node.value}, Probs: {node.probs}\n"

        result = f"{'|   ' * depth}Feature {node.feature}"
        if node.threshold is not None:
            result += f" < {node.threshold}"
        result += "\n"

        for child_key, child_node in node.children.items():
            result += f"{'|   ' * (depth + 1)}Child '{child_key}' (Prob: {node.children_probs[child_key]}):\n"
            result += self._print_tree(child_node, depth + 2)

        return result

    def visualize(self, filename="decision_tree", save_to_file=False):
            if self.root is None:
                raise ValueError("Model has not been fitted yet.")
    
            positions = {}
            num_leaves, max_depth = self._compute_tree_layout(self.root, positions=positions)
    
            fig_width = max(8.0, num_leaves * 2.6)
            fig_height = max(5.0, (max_depth + 1) * 2.3)
    
            rc_style = {
                "font.family": ["Arial", "sans-serif"],
                "font.size": 11,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "axes.linewidth": 2.0,
                "legend.frameon": False,
                "svg.fonttype": "none",
            }
    
            with plt.rc_context(rc_style):
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.set_axis_off()
                ax.set_xlim(-0.7, max(0.7, (num_leaves - 1) + 0.7))
                ax.set_ylim(-0.6, max(0.6, max_depth + 0.6))
    
                self._draw_tree_nodes(ax, self.root, positions, max_depth)
                fig.tight_layout(pad=1.5)
    
                if save_to_file:
                    base, ext = os.path.splitext(filename)
                    parent_dir = os.path.dirname(filename)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
    
                    if ext:
                        fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.2)
                        if ext.lower() != ".png":
                            fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
                    else:
                        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    
                plt.close(fig)
                return fig

    def _compute_tree_layout(self, node, depth=0, next_leaf_x=0, positions=None):
        if positions is None:
            positions = {}

        if node.is_leaf or not node.children:
            positions[id(node)] = (float(next_leaf_x), depth)
            return next_leaf_x + 1, depth

        child_xs = []
        max_depth = depth
        for child_node in node.children.values():
            next_leaf_x, child_max_depth = self._compute_tree_layout(
                child_node, depth + 1, next_leaf_x, positions
            )
            child_xs.append(positions[id(child_node)][0])
            max_depth = max(max_depth, child_max_depth)

        positions[id(node)] = (0.5 * (child_xs[0] + child_xs[-1]), depth)
        return next_leaf_x, max_depth

    def _format_leaf_label(self, node):
        if self.classes is not None and node.probs is not None:
            prob_parts = [f"{cls}: {prob:.2f}" for cls, prob in zip(self.classes, node.probs)]
            lines = [", ".join(prob_parts[i:i + 2]) for i in range(0, len(prob_parts), 2)]
            return f"Leaf: {node.value}\n" + "\n".join(lines)
        return f"Leaf: {node.value}"

    @staticmethod
    def _format_edge_label(node, child_key):
        prob = node.children_probs[child_key]
        if node.threshold is not None:
            op = "<" if child_key == "left" else "≥"
            cond = f"{op} {node.threshold:.4g}"
        else:
            cond = f"= {child_key}"
        return f"{cond}\n(p={prob:.2f})"

    def _draw_tree_nodes(self, ax, node, positions, max_depth):
        nx, ndepth = positions[id(node)]
        ny = max_depth - ndepth

        if not node.is_leaf and node.children:
            num_children = len(node.children)
            for idx, (child_key, child_node) in enumerate(node.children.items()):
                cx, cdepth = positions[id(child_node)]
                cy = max_depth - cdepth

                ax.plot([nx, cx], [ny, cy], color="#4D4D4D", linewidth=1.8, zorder=1)
                
                t = 0.55 if num_children <= 2 else (0.50 if idx % 2 == 0 else 0.68)
                lx = nx + t * (cx - nx)
                ly = ny + t * (cy - ny)
                ax.text(
                    lx,
                    ly,
                    self._format_edge_label(node, child_key),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="#272727",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.92),
                    zorder=2,
                )

                self._draw_tree_nodes(ax, child_node, positions, max_depth)

        if node.is_leaf:
            label = self._format_leaf_label(node)
            bbox = dict(boxstyle="round,pad=0.45", facecolor="#DDF3DE", edgecolor="#42949E", linewidth=1.8)
            weight = "normal"
        else:
            label = f"Feature {node.feature}"
            bbox = dict(boxstyle="round,pad=0.45", facecolor="#EEF4FB", edgecolor="#0F4D92", linewidth=2.0)
            weight = "bold"

        ax.text(
            nx,
            ny,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight=weight,
            color="#272727",
            bbox=bbox,
            zorder=3,
        )