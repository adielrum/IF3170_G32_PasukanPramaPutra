"""
Decision Tree Learning (ID3 / C4.5 / CART) classifier from scratch.

Supports categorical and continuous features, fractional sample weighting for
missing values (C4.5), vectorized prefix-sum continuous threshold search, and
tree visualization with top-N depth truncation.

References:
- N. U. Maulidevi, "Decision Tree" & "Issues in Decision Tree," IF3170
  Inteligensi Artifisial, Edunex ITB, 2026.
- J. R. Quinlan, "C4.5: Programs for Machine Learning," Morgan Kaufmann, 1993
  (Gain Ratio, fractional sample weights for missing attribute values).
- J. R. Quinlan, "Improved Use of Continuous Attributes in C4.5," Journal of
  Artificial Intelligence Research (JAIR), vol. 4, pp. 77-90, 1996
  (mean-gain threshold filtering for continuous splits).
- L. Breiman et al., "Classification and Regression Trees," Wadsworth, 1984
  (Gini impurity criterion & Minimal Cost-Complexity Pruning / ccp_alpha).
"""

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
        impurity=0.0,
        weight=0.0,
        majority_value=None,
        majority_probs=None,
    ):
        self.feature = feature
        self.children = children 
        self.children_probs = children_probs 
        self.threshold = threshold
        self.value = value
        self.probs = probs
        self.impurity = impurity
        self.weight = weight
        self.majority_value = majority_value
        self.majority_probs = majority_probs

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
        ccp_alpha=0.0,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Invalid criterion: {criterion}")
        if ccp_alpha < 0.0:
            raise ValueError(f"ccp_alpha must be non-negative, got {ccp_alpha}.")

        self.criterion = criterion
        self.use_gain_ratio = bool(use_gain_ratio)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features
        self.ccp_alpha = float(ccp_alpha)

        self.root = None
        self.classes = None
        self.categories = {}
        self.feature_names = None

    def fit(self, X, y):

        # Convert categorical_features to indices
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
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

        if self.ccp_alpha > 0.0 and self.root is not None:
            self._prune_ccp(self.root, total_weight=float(np.sum(initial_weights)))

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
        total_counts = float(np.sum(weighted_counts))

        probs = weighted_counts / total_counts 
        majority = self.classes[np.argmax(weighted_counts)]
        impurity = float(self._loss_from_probs(probs))

        return Node(value=majority, probs=probs, impurity=impurity, weight=total_counts)

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

        weighted_counts = np.bincount(y_indices, weights=weights, minlength=len(self.classes))
        node_probs = weighted_counts / total_weight
        node_majority = self.classes[np.argmax(weighted_counts)]
        node_impurity = float(self._loss_from_probs(node_probs))

        return Node(
            feature=best_feature,
            children=children,
            children_probs=children_probs,
            threshold=best_threshold,
            impurity=node_impurity,
            weight=float(total_weight),
            majority_value=node_majority,
            majority_probs=node_probs,
        )

    def _prune_ccp(self, root, total_weight):
        """Minimal Cost-Complexity Pruning (weakest-link post-pruning).

        Iteratively collapses the internal node with the smallest effective
        alpha = (R(t) - R(T_t)) / (|T_t| - 1) while alpha <= self.ccp_alpha,
        where R(t) = (w_t / W) * impurity(t).

        Reference:
        L. Breiman, J. H. Friedman, R. A. Olshen, & C. J. Stone,
        "Classification and Regression Trees (CART)," Wadsworth, 1984.
        """
        if total_weight <= 0.0:
            return

        while not root.is_leaf:
            best_node = None
            best_alpha = float("inf")

            def traverse(node):
                nonlocal best_node, best_alpha
                r_node = (node.weight / total_weight) * node.impurity
                if node.is_leaf or not node.children:
                    return r_node, 1

                r_subtree = 0.0
                n_leaves = 0
                for child in node.children.values():
                    r_c, l_c = traverse(child)
                    r_subtree += r_c
                    n_leaves += l_c

                if n_leaves > 1:
                    alpha_eff = max(0.0, (r_node - r_subtree) / (n_leaves - 1))
                    if alpha_eff < best_alpha:
                        best_alpha = alpha_eff
                        best_node = node

                return r_subtree, n_leaves

            traverse(root)

            if best_node is None or best_alpha > self.ccp_alpha:
                break

            # Collapse weakest link into a leaf node
            best_node.value = best_node.majority_value
            best_node.probs = best_node.majority_probs
            best_node.children = None
            best_node.children_probs = None
            best_node.feature = None
            best_node.threshold = None

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

    def _subtree_probs(self, node):
        if node.is_leaf or not node.children:
            return np.asarray(node.probs, dtype=np.float64)
        probs = np.zeros(len(self.classes), dtype=np.float64)
        for key, child in node.children.items():
            p = (node.children_probs or {}).get(key, 0.0)
            if p > 0.0:
                probs += p * self._subtree_probs(child)
        total = probs.sum()
        return probs / total if total > 0 else probs

    def visualize_tree(self, filename="decision_tree.png", save_to_file=False, top_n=None, feature_names=None):
        if self.root is None:
            raise ValueError("Model has not been fitted yet.")
        if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 0):
            raise ValueError("top_n must be a non-negative integer or None.")

        names = feature_names if feature_names is not None else getattr(self, "feature_names", None)
        pos, next_x, max_d = {}, [0.0], [0]

        def layout(node, d=0):
            max_d[0] = max(max_d[0], d)
            if node.is_leaf or not node.children or (top_n is not None and d >= top_n):
                pos[id(node)] = (next_x[0], d)
                next_x[0] += 1.0
                return pos[id(node)][0]
            xs = [layout(c, d + 1) for c in node.children.values()]
            pos[id(node)] = (0.5 * (xs[0] + xs[-1]), d)
            return pos[id(node)][0]

        layout(self.root)
        n_leaves, max_depth = max(1, int(next_x[0])), max_d[0]
        fig, ax = plt.subplots(figsize=(max(6, n_leaves * 2.2), max(4, (max_depth + 1) * 2.0)))
        ax.set_axis_off()

        def draw(node, d=0):
            x, y = pos[id(node)][0], max_depth - d
            truncated = not node.is_leaf and bool(node.children) and top_n is not None and d >= top_n
            if not node.is_leaf and node.children and not truncated:
                n_ch = len(node.children)
                for i, (key, child) in enumerate(node.children.items()):
                    cx, cy = pos[id(child)][0], max_depth - (d + 1)
                    ax.plot([x, cx], [y, cy], "k-", lw=1)
                    t = 0.5 if n_ch <= 2 else (0.45 if i % 2 == 0 else 0.65)
                    cond = f"{'<' if key == 'left' else '>='} {node.threshold:.3g}" if node.threshold is not None else f"= {key}"
                    ax.text(x + t * (cx - x), y + t * (cy - y), f"{cond}\n({node.children_probs[key]:.2f})",
                            ha="center", va="center", fontsize=8, bbox=dict(fc="white", ec="none"))
                    draw(child, d + 1)

            if node.is_leaf or not node.children:
                lbl, fc = f"Leaf: {node.value}", "lightgreen"
            elif truncated:
                pred = self.classes[int(np.argmax(self._subtree_probs(node)))]
                lbl, fc = f"...\n(pred: {pred})", "lightgray"
            else:
                feat = names[node.feature] if names is not None and node.feature < len(names) else f"Feature {node.feature}"
                lbl, fc = feat, "lightblue"
            ax.text(x, y, lbl, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc=fc, ec="black"))

        draw(self.root)
        ax.set_xlim(-0.8, max(0.8, n_leaves - 0.2))
        ax.set_ylim(-0.6, max_depth + 0.6)
        plt.tight_layout()
        if save_to_file:
            if not os.path.splitext(filename)[1]:
                filename += ".png"
            if parent := os.path.dirname(filename):
                os.makedirs(parent, exist_ok=True)
            fig.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return fig