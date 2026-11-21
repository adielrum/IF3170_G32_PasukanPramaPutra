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
        *, 
        max_depth=100,
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_features=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Invalid criterion: {criterion}")

        self.criterion = criterion
        self.use_gain_ratio = bool(use_gain_ratio)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features

        self.root = None
        self.classes = None
        self.categories = {}

    def fit(self, X, y):

        # Convert categorical_features to indices
        if hasattr(X, 'columns') and self.categorical_features:
            col_list = list(X.columns)
            self.categorical_features = [
                col_list.index(c) if isinstance(c, str) else c 
                for c in self.categorical_features
            ]

        # 
        X_clean, is_num = self._prepare_X(X) 
        self.all_numeric_features = is_num 
        y = np.asarray(y)

        self.classes, y_indices = np.unique(y, return_inverse=True)
        self.categories = self._infer_feature_types(X_clean)

        initial_weights = np.ones(len(y_indices), dtype=float)
        self.root = self._build_tree(X_clean, y_indices, initial_weights, depth=0)

        return self
    
    @staticmethod
    def _is_null_value(x):
        return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))

    @staticmethod
    def _is_num_value(x):
        return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))

    @classmethod
    def _is_null(cls, arr):

        # Numeric feature 
        if np.issubdtype(arr.dtype, np.number):
            return np.isnan(arr)

        # Categorical feature 
        else: 
            return np.array([cls._is_null_value(x) for x in arr], dtype=bool)

    def _prepare_X(self, X):
        X_arr = np.asarray(X)

        # All numeric features 
        if np.issubdtype(X_arr.dtype, np.number) and not self.categorical_features:
            return np.ascontiguousarray(X_arr, dtype=np.float64), True

        # Contains categorical features 
        else: 
            return np.asarray(X, dtype=object), False

    def _infer_feature_types(self, X):
        if self.all_numeric_features:
            return {}
        
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
        min_split_weight = max(self.min_samples_split, 2 * self.min_samples_leaf)
        if len(unique_labels) <= 1:
            return self._create_leaf(y_indices, weights)
        if total_weight < min_split_weight or depth >= self.max_depth:
            return self._create_leaf(y_indices, weights)

        # Select best split
        best_feature, best_threshold = self._best_split(X, y_indices, weights)

        if best_feature is None:
            return self._create_leaf(y_indices, weights)

        # Separate known and unknown values 
        col = X[:, best_feature]
        null_mask = self._is_null(col)
        known_mask = ~null_mask
        known_weight = np.sum(weights[known_mask])

        # Categorical feature branching
        if best_feature in self.categories:
            child_masks = {
                val: known_mask & (col == val)
                for val in self.categories[best_feature]
            }

        # Continuous feature branching
        else:
            if best_threshold is None:
                raise ValueError(
                    f"Continuous feature {best_feature} returned threshold=None."
                )

            if self.all_numeric_features:
                child_masks = {
                    "left": known_mask & (col < best_threshold),
                    "right": known_mask & (col >= best_threshold),
                }
            else:
                col_known = np.zeros(len(X), dtype=np.float64)
                col_known[known_mask] = col[known_mask].astype(np.float64)
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

    def _loss_from_probs(self, probs, axis=-1):
        probs = np.asarray(probs, dtype=np.float64)
        safe_probs = np.where(probs > 0.0, probs, 1.0)

        match self.criterion:
            case "entropy":
                imp = -np.sum(np.where(probs > 0.0, probs * np.log2(safe_probs), 0.0), axis=axis)
            
            case "gini":
                imp = 1.0 - np.sum(probs ** 2, axis=axis)
            
            case _:
                raise ValueError(f"Invalid criterion: {self.criterion}.")

        return float(imp) if np.ndim(imp) == 0 else imp

    def _loss(self, y_indices, weights):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        total_weight = np.sum(weights)
        counts = np.bincount(y_indices, weights=weights, minlength=len(self.classes))
        probs = counts[counts > 0] / total_weight

        return self._loss_from_probs(probs)

    def _gain(
        self,
        parent_probs,
        child_class_weights,
        W_known,
        w_unknown=0.0,
        *,
        filter_mean_gain=False,
    ):
        child_class_weights = np.asarray(child_class_weights, dtype=np.float64)
        S = child_class_weights.shape[0]
        W_tot = W_known + w_unknown

        # L(S)
        parent_imp = self._loss_from_probs(parent_probs)

        # |Sᵥ|/|S| for every v
        child_weights = np.sum(child_class_weights, axis=-1)
        safe_child_weights = np.where(child_weights > 0.0, child_weights, 1.0)

        # L(Sᵥ) for every v
        child_probs = child_class_weights / safe_child_weights[..., None]
        child_imps = self._loss_from_probs(child_probs, axis=-1) 

        # Σᵥ |Sᵥ|/|S| L(Sᵥ) 
        weighted_child_imp = np.sum((child_weights / W_known) * child_imps, axis=-1)

        # Gain 
        gains = (W_known / W_tot) * (parent_imp - weighted_child_imp)
        if not self.use_gain_ratio:
            return np.maximum(gains, 0.0)

        # Split loss 
        p_parts = [child_weights / W_tot]
        if w_unknown > 0.0:
            p_parts.append(np.full((S, 1), w_unknown / W_tot, dtype=np.float64))
        split_imp = self._loss_from_probs(np.concatenate(p_parts, axis=-1), axis=-1)

        valid = (gains > 1e-12) & (split_imp > 1e-12)
        if filter_mean_gain and np.any(gains > 1e-12):
            mean_gain = float(np.mean(gains[gains > 1e-12]))
            valid &= gains >= mean_gain - 1e-12

        # Gain Ratio 
        safe_split_imp = np.where(valid, split_imp, 1.0)
        gain_ratios = np.where(valid, gains / safe_split_imp, 0.0)
        return gain_ratios

    def _best_continuous_split(self, col_known, y_known, w_known, w_unknown):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")

        pos_w = w_known > 0.0
        col_known, y_known, w_known = col_known[pos_w], y_known[pos_w], w_known[pos_w]

        m = len(col_known)
        if m < 2:
            return 0.0, None

        # Sort vavlues to candidate thresholds  
        order = np.argsort(col_known, kind="mergesort")
        x_s, y_s, w_s = col_known[order], y_known[order], w_known[order]

        valid_splits = np.flatnonzero(x_s[:-1] < x_s[1:])
        if len(valid_splits) == 0:
            return 0.0, None

        # Precompute weights at each possible splits 
        W_mat = np.zeros((m, len(self.classes)), dtype=np.float64)
        W_mat[np.arange(m), y_s] = w_s

        cum_W = np.cumsum(W_mat[:-1], axis=0)
        total_W_class = cum_W[-1] + W_mat[-1]
        W_known = float(np.sum(total_W_class))

        left_W = cum_W[valid_splits]
        right_W = total_W_class - left_W
        w_left = np.sum(left_W, axis=1)
        w_right = W_known - w_left

        # Enforce min_samples_leaf 
        min_leaf = max(1e-12, float(self.min_samples_leaf))
        feasible = (w_left >= min_leaf) & (w_right >= min_leaf)
        if not np.any(feasible):
            return 0.0, None

        valid_splits = valid_splits[feasible]
        w_left, w_right = w_left[feasible], w_right[feasible]

        child_class_weights = np.stack([left_W[feasible], right_W[feasible]], axis=1)
        parent_probs = total_W_class / W_known

        scores = self._gain(
            parent_probs,
            child_class_weights,
            W_known,
            w_unknown,
            filter_mean_gain=True,
        )

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score <= 0.0:
            return 0.0, None

        k = int(valid_splits[best_idx])
        return best_score, float(0.5 * (x_s[k] + x_s[k + 1]))

    def _best_split(self, X, y_indices, weights):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")

        num_features = X.shape[1]
        best_score = 0.0
        best_feature = None
        best_threshold = None

        for j in range(num_features):
            col = X[:, j]
            null_mask = self._is_null(col)
            known_mask = ~null_mask

            w_known = weights[known_mask]
            if np.sum(w_known) <= 0.0:
                continue

            y_known = y_indices[known_mask]
            w_unknown = np.sum(weights[null_mask])

            # Categorical feature
            if j in self.categories:
                col_known = col[known_mask]
                cat_vals = self.categories[j]
                K = len(self.classes)

                branch_W = np.array([
                    np.bincount(
                        y_known[col_known == val],
                        weights=w_known[col_known == val],
                        minlength=K,
                    )
                    for val in cat_vals
                ], dtype=np.float64,)[None, :, :]

                if np.sum(np.sum(branch_W[0], axis=-1) >= self.min_samples_leaf) < 2:
                    continue

                W_known = float(np.sum(w_known))
                parent_probs = np.sum(branch_W[0], axis=0) / W_known
                score = float(self._gain(parent_probs, branch_W, W_known, w_unknown)[0])
                threshold = None

            # Continuous feature
            else:
                col_known = (
                    col[known_mask]
                    if self.all_numeric_features
                    else col[known_mask].astype(np.float64)
                )
                score, threshold = self._best_continuous_split(
                    col_known, y_known, w_known, w_unknown
                )

            if score > best_score:
                best_score = score
                best_feature = j
                best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        if self.classes is None:
            raise ValueError("Model has not been fitted yet.")
        
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]
    
    def _predict_batch(self, X, row_indices, node, out, is_numeric, weight=1.0):
        if len(row_indices) == 0:
            return

        if node.is_leaf:
            out[row_indices] += weight * node.probs
            return

        col = X[row_indices, node.feature]
        null_mask = self._is_null(col)

        # Missing values
        if np.any(null_mask):
            null_rows = row_indices[null_mask]
            for child_key, child_node in node.children.items():
                p_child = node.children_probs[child_key]
                if p_child > 0.0:
                    self._predict_batch(X, null_rows, child_node, out, is_numeric, weight * p_child)

        known_mask = ~null_mask
        if not np.any(known_mask):
            return

        known_rows = row_indices[known_mask]
        known_vals = col[known_mask]

        # Continuous split 
        if node.threshold is not None:
            vals_float = (
                known_vals if is_numeric else known_vals.astype(np.float64)
            )
            go_left = vals_float < node.threshold
            self._predict_batch(X, known_rows[go_left], node.children["left"], out, is_numeric, weight)
            self._predict_batch(X, known_rows[~go_left], node.children["right"], out, is_numeric, weight)
            return

        # Categorical split 
        unseen_mask = np.ones(len(known_rows), dtype=bool)
        for child_key, child_node in node.children.items():
            match_mask = known_vals == child_key
            if np.any(match_mask):
                unseen_mask &= ~match_mask
                self._predict_batch(
                    X, known_rows[match_mask], child_node, out, is_numeric, weight
                )

        if np.any(unseen_mask):
            unseen_rows = known_rows[unseen_mask]
            for child_key, child_node in node.children.items():
                p_child = node.children_probs[child_key]
                if p_child > 0.0:
                    self._predict_batch(X, unseen_rows, child_node, out, is_numeric, weight * p_child)

    def predict_proba(self, X):
        if self.root is None or self.classes is None:
            raise ValueError("Model has not been fitted yet.")

        X_clean, is_num = self._prepare_X(X)
        n_samples = len(X_clean)
        out = np.zeros((n_samples, len(self.classes)), dtype=np.float64)
        if n_samples == 0:
            return out

        self._predict_batch(X_clean, np.arange(n_samples), self.root, out, is_num, weight=1.0)
        return out

    def print_tree(self):
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

    def visualize_tree(self, filename="decision_tree", save_to_file=False):
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