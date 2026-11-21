"""
Naive Bayes classifier from scratch with quantile binning for continuous
features, Laplace (additive) smoothing, and missing value handling.

References:
- N. U. Maulidevi, "Bayes Net," IF3170 Inteligensi Artifisial, Edunex ITB, 2024.
- T. M. Mitchell, "Machine Learning," McGraw-Hill, 1997 (Naive Bayes &
  Laplace/m-estimate smoothing).
- C. D. Manning, P. Raghavan, and H. Schutze, "Introduction to Information
  Retrieval," Cambridge University Press, 2008, Ch. 13 (log-space Naive Bayes
  with additive smoothing).
"""

import os
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel


_NB_PALETTE = {
    "blue_main": "#0F4D92",       # Class variable / primary Naive Bayes edges
    "blue_secondary": "#3775BA",  # Numeric (quantile-binned) feature nodes
    "teal": "#42949E",            # Auxiliary accent
    "green_1": "#DDF3DE",         # Light categorical fill
    "green_2": "#AADCA9",         # Categorical / binary feature nodes
    "red_strong": "#B64342",      # Empirical inter-feature dependency edges
    "neutral": "#CFCECE",         # Light neutral fill / subtle edge
    "text_dark": "#111827",       # Crisp dark border & text
    "text_muted": "#4B5563",      # Secondary annotations
}


@contextmanager
def _nb_style_context():
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.linewidth": 2.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }):
        yield


def _save_nb_figure(fig, filename, formats=("png",), dpi=300):
    base, ext = os.path.splitext(filename)
    targets = list(formats) if formats else [ext.lstrip(".") or "png"]
    if ext and ext.lstrip(".").lower() not in [f.lower() for f in targets]:
        targets.insert(0, ext.lstrip("."))
    if parent := os.path.dirname(base):
        os.makedirs(parent, exist_ok=True)
    for fmt in targets:
        out_path = f"{base}.{fmt.lstrip('.')}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")


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

    @staticmethod
    def _mi_from_contingency(table: np.ndarray) -> float:
        total = float(np.sum(table))
        if total <= 0.0:
            return 0.0
        p_xy = table / total
        p_x = np.sum(p_xy, axis=1, keepdims=True)
        p_y = np.sum(p_xy, axis=0, keepdims=True)
        denom = p_x @ p_y
        mask = (p_xy > 0.0) & (denom > 0.0)
        return float(np.sum(p_xy[mask] * np.log2(p_xy[mask] / denom[mask])))

    def _compute_class_feature_mi(self) -> np.ndarray:
        if not hasattr(self, "feature_count_") or not hasattr(self, "class_count_"):
            raise ValueError("Model has not been fitted yet.")

        n_features = len(self.feature_names_)
        n_classes = len(self.classes_)
        mi_scores = np.zeros(n_features, dtype=np.float64)

        for j in range(n_features):
            values = list(self.feature_values_[j])
            if not values:
                continue
            table = np.zeros((len(values), n_classes), dtype=np.float64)
            for v_idx, val in enumerate(values):
                for c_idx in range(n_classes):
                    table[v_idx, c_idx] = self.feature_count_[j][c_idx].get(val, 0.0)
            mi_scores[j] = self._mi_from_contingency(table)

        return mi_scores

    def _compute_feature_pair_mi(self, X, feature_indices, max_edges=10, min_mi=0.02):
        X_arr = self.prep_X(X)
        if len(X_arr) == 0 or len(feature_indices) < 2:
            return []

        encoded_cols = []
        cardinalities = []
        for j in feature_indices:
            feat = self.feature_names_[j]
            val_to_code = {}
            codes = np.empty(len(X_arr), dtype=np.int64)
            for row_idx, raw_val in enumerate(X_arr[:, j]):
                t_val = self.transform(feat, raw_val)
                code = val_to_code.get(t_val)
                if code is None:
                    code = len(val_to_code)
                    val_to_code[t_val] = code
                codes[row_idx] = code
            encoded_cols.append(codes)
            cardinalities.append(len(val_to_code))

        edges = []
        for a in range(len(feature_indices)):
            ca, ua = encoded_cols[a], cardinalities[a]
            if ua < 2:
                continue
            for b in range(a + 1, len(feature_indices)):
                cb, ub = encoded_cols[b], cardinalities[b]
                if ub < 2:
                    continue
                table = np.bincount(ca * ub + cb, minlength=ua * ub).reshape(ua, ub)
                mi = self._mi_from_contingency(table)
                if mi >= min_mi:
                    edges.append((feature_indices[a], feature_indices[b], mi))

        edges.sort(key=lambda item: item[2], reverse=True)
        return edges[:max_edges]

    @staticmethod
    def _order_features_for_ring(indices, pair_edges):
        m = len(indices)
        if m <= 3 or not pair_edges:
            return list(indices)

        idx_set = set(indices)
        edges = [(u, v, w) for u, v, w in pair_edges if u in idx_set and v in idx_set]
        if not edges:
            return list(indices)

        adj = {idx: {} for idx in indices}
        for u, v, w in edges:
            adj[u][v] = adj[v][u] = w

        # Initial greedy ordering with connected-component DFS so components stay contiguous
        visited = set()
        order = []
        nodes_by_strength = sorted(indices, key=lambda idx: (-sum(adj[idx].values()), idx))
        for seed in nodes_by_strength:
            if seed in visited:
                continue
            cur = seed
            while cur is not None:
                order.append(cur)
                visited.add(cur)
                cands = [n for n in adj[cur] if n not in visited]
                if cands:
                    cur = max(cands, key=lambda n: (adj[cur][n], -n))
                else:
                    # Stay within same connected component if any unvisited neighbor exists
                    comp_cands = [
                        n for n in indices
                        if n not in visited and any(n in adj[u] for u in order)
                    ]
                    cur = max(comp_cands, key=lambda n: sum(adj[n].values())) if comp_cands else None

        # Refine circular order with 2-opt / swap & segment-shift passes
        def ring_cost(seq):
            pos = {node: k for k, node in enumerate(seq)}
            cost = 0.0
            for u, v, w in edges:
                diff = abs(pos[u] - pos[v])
                ang_step = min(diff, m - diff)
                cost += w * (ang_step ** 2)
            return cost

        best_cost = ring_cost(order)
        improved = True
        while improved:
            improved = False
            for i in range(m):
                for j in range(i + 1, m):
                    cand = order[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c_val = ring_cost(cand)
                    if c_val + 1e-12 < best_cost:
                        order = cand
                        best_cost = c_val
                        improved = True

        return order

    def visualize_graph(
        self,
        filename="naive_bayes_graph.png",
        save_to_file=False,
        *,
        X=None,
        target_name="is_canceled",
        feature_names=None,
        top_n=None,
        show_feature_edges=True,
        max_feature_edges=8,
        min_feature_mi=0.02,
        formats=("png",),
        dpi=300,
    ):
        if not hasattr(self, "class_prior_") or not hasattr(self, "feature_names_"):
            raise ValueError("Model has not been fitted yet.")
        if top_n is not None and (
            isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1
        ):
            raise ValueError("top_n must be a positive integer or None.")

        n_features = len(self.feature_names_)
        display_names = (
            list(feature_names)
            if feature_names is not None
            else [
                str(f) if not isinstance(f, (int, np.integer)) else f"Feature {f}"
                for f in self.feature_names_
            ]
        )

        mi_class = self._compute_class_feature_mi()
        order_by_mi = np.argsort(mi_class)[::-1]
        if top_n is not None and top_n < n_features:
            selected = [int(i) for i in order_by_mi[: int(top_n)]]
        else:
            selected = list(range(n_features))

        pair_edges = []
        if X is not None and show_feature_edges and len(selected) >= 2:
            pair_edges = self._compute_feature_pair_mi(
                X, selected, max_edges=max_feature_edges, min_mi=min_feature_mi
            )

        ring_order = self._order_features_for_ring(selected, pair_edges)
        m = len(ring_order)

        rx = max(8.8, 0.50 * m)
        ry = max(5.4, 0.31 * m)
        angles = np.linspace(0.5 * np.pi, 0.5 * np.pi - 2.0 * np.pi, m, endpoint=False)
        node_pos = {
            idx: (rx * float(np.cos(ang)), ry * float(np.sin(ang)))
            for idx, ang in zip(ring_order, angles)
        }
        ring_idx = {idx: k for k, idx in enumerate(ring_order)}

        max_mi_c = max(float(mi_class[selected].max()), 1e-6) if selected else 1.0
        max_mi_f = max((w for _, _, w in pair_edges), default=1.0)

        with _nb_style_context():
            fig, ax = plt.subplots(figsize=(16.0, 10.5))
            ax.set_axis_off()

            # 1. Empirical inter-feature edges routed strictly outside the perimeter
            pos_list = np.array([node_pos[idx] for idx in ring_order])
            min_lx, max_lx = -rx, rx
            min_ly, max_ly = -ry, ry

            for u, v, mi_uv in pair_edges:
                x1, y1 = node_pos[u]
                x2, y2 = node_pos[v]
                dx, dy = x2 - x1, y2 - y1
                mid_x, mid_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                # Always bend outward away from (0, 0):
                # matplotlib arc3 places control point at mid + rad * (dy, -dx)
                dot_out = dy * mid_x + (-dx) * mid_y
                sign = 1.0 if dot_out >= 0.0 else -1.0

                step_diff = abs(ring_idx[u] - ring_idx[v])
                step_span = min(step_diff, m - step_diff)
                # Higher arc for adjacent nodes so it clears node boxes; gentler for multi-hop
                mag = 0.58 if step_span == 1 else (0.42 if step_span == 2 else 0.30)

                # Ensure Bezier midpoint clears all nearby feature boxes
                for _ in range(5):
                    rad = sign * mag
                    lx = mid_x + 0.5 * rad * dy
                    ly = mid_y - 0.5 * rad * dx
                    dists = np.hypot((pos_list[:, 0] - lx) / 1.45, (pos_list[:, 1] - ly) / 0.72)
                    if np.min(dists) >= 1.05:
                        break
                    mag += 0.09

                rad = sign * mag
                lx = mid_x + 0.5 * rad * dy
                ly = mid_y - 0.5 * rad * dx
                min_lx, max_lx = min(min_lx, lx), max(max_lx, lx)
                min_ly, max_ly = min(min_ly, ly), max(max_ly, ly)

                lw = 1.2 + 2.0 * (mi_uv / max_mi_f)
                patch = mpatches.FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    connectionstyle=f"arc3,rad={rad:.3f}",
                    arrowstyle="-",
                    linestyle="--",
                    linewidth=lw,
                    color=_NB_PALETTE["red_strong"],
                    alpha=0.78,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=2,
                )
                ax.add_patch(patch)

                ax.text(
                    lx,
                    ly,
                    f"{mi_uv:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    color=_NB_PALETTE["red_strong"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92),
                    zorder=4,
                )

            # 2. Naive Bayes directed edges: Class -> Feature
            for pos_idx, feat_idx in enumerate(ring_order):
                fx, fy = node_pos[feat_idx]
                mi_val = float(mi_class[feat_idx])
                ratio = mi_val / max_mi_c
                lw = 1.2 + 2.8 * ratio
                alpha = 0.48 + 0.48 * ratio

                # Compute exact ray intersection with approximate feature box half-sizes
                name = display_names[feat_idx]
                box_hw = max(0.85, 0.056 * len(name) + 0.34)
                box_hh = 0.42
                tx = box_hw / max(abs(fx), 1e-9)
                ty = box_hh / max(abs(fy), 1e-9)
                t_hit = min(tx, ty, 0.45)
                end_x = fx * (1.0 - t_hit)
                end_y = fy * (1.0 - t_hit)

                arrow = mpatches.FancyArrowPatch(
                    (0.0, 0.0),
                    (end_x, end_y),
                    arrowstyle="-|>,head_length=7.5,head_width=4.4",
                    linewidth=lw,
                    color=_NB_PALETTE["blue_main"],
                    alpha=alpha,
                    shrinkA=0,
                    shrinkB=2,
                    zorder=3,
                )
                ax.add_patch(arrow)

                # Stagger MI labels along spokes to prevent radial crowding near center
                t_lbl = 0.50 if pos_idx % 2 == 0 else 0.65
                ax.text(
                    t_lbl * fx,
                    t_lbl * fy,
                    f"{mi_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    color=_NB_PALETTE["blue_main"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92),
                    zorder=4,
                )

            # 3. Feature nodes
            for feat_idx in ring_order:
                fx, fy = node_pos[feat_idx]
                raw_feat = self.feature_names_[feat_idx]
                is_num = raw_feat in self.numeric_features_
                if is_num:
                    n_bins_eff = len(self.bin_edges_.get(raw_feat, [])) + 1
                    meta = f"num ({n_bins_eff} bins)"
                    fc = _NB_PALETTE["blue_secondary"]
                    tc = "white"
                else:
                    n_cats = len(self.feature_values_[feat_idx])
                    meta = f"cat ({n_cats} vals)"
                    fc = _NB_PALETTE["green_2"]
                    tc = _NB_PALETTE["text_dark"]

                name = display_names[feat_idx]
                label = f"{name}\n[{meta}]"
                ax.text(
                    fx,
                    fy,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    fontweight="normal",
                    color=tc,
                    bbox=dict(
                        boxstyle="round,pad=0.42",
                        fc=fc,
                        ec=_NB_PALETTE["text_dark"],
                        lw=1.4,
                    ),
                    zorder=6,
                )

            # 4. Central Class Node
            prior_parts = [
                f"P({cls})={p:.2f}" for cls, p in zip(self.classes_, self.class_prior_)
            ]
            class_label = f"{target_name}\n" + " | ".join(prior_parts)
            ax.text(
                0.0,
                0.0,
                class_label,
                ha="center",
                va="center",
                fontsize=11.0,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.58",
                    fc=_NB_PALETTE["blue_main"],
                    ec=_NB_PALETTE["text_dark"],
                    lw=2.0,
                ),
                zorder=7,
            )

            pad_x = 2.4
            pad_y = 1.6
            ax.set_xlim(min(-rx, min_lx) - pad_x, max(rx, max_lx) + pad_x)
            ax.set_ylim(min(-ry, min_ly) - pad_y, max(ry, max_ly) + pad_y)

            title_suffix = (
                f"Top-{m} dari {n_features} Fitur"
                if top_n is not None and top_n < n_features
                else f"{m} Fitur"
            )
            ax.set_title(
                f"Naive Bayes Bayesian Network & Dependensi Fitur ({title_suffix})\n"
                "Bobot Sisi: Mutual Information $I(\\cdot\\,;\\,\\cdot)$ dalam bit",
                pad=14,
            )

            legend_handles = [
                mpatches.Patch(
                    facecolor=_NB_PALETTE["blue_main"],
                    edgecolor=_NB_PALETTE["text_dark"],
                    linewidth=1.5,
                    label=f"Variabel Kelas ({target_name})",
                ),
                mpatches.Patch(
                    facecolor=_NB_PALETTE["blue_secondary"],
                    edgecolor=_NB_PALETTE["text_dark"],
                    linewidth=1.2,
                    label=f"Fitur Numerik (Quantile Binning, n_bins={self.n_bins})",
                ),
                mpatches.Patch(
                    facecolor=_NB_PALETTE["green_2"],
                    edgecolor=_NB_PALETTE["text_dark"],
                    linewidth=1.2,
                    label="Fitur Kategorikal / Biner",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color=_NB_PALETTE["blue_main"],
                    linewidth=2.4,
                    marker=">",
                    markersize=7,
                    label=r"Ketergantungan Model Naive Bayes: $C \rightarrow X_j$",
                ),
            ]
            if pair_edges:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=_NB_PALETTE["red_strong"],
                        linestyle="--",
                        linewidth=2.0,
                        label=(
                            f"Dependensi Empiris Antarfitur (Top-{len(pair_edges)}, "
                            "diabaikan oleh asumsi independensi Naive Bayes)"
                        ),
                    )
                )

            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=2,
                frameon=False,
                fontsize=10.0,
                columnspacing=1.8,
                handletextpad=0.6,
            )

            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
            if save_to_file:
                _save_nb_figure(fig, filename, formats=formats, dpi=dpi)
            plt.close(fig)
            return fig
