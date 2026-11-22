"""
Support Vector Machine (SVC / BinarySVC) classifier from scratch.

Trained using Sequential Minimal Optimization (SMO) with Second-Order Working
Set Selection (WSS2) and an LRU kernel-column cache (LIBSVM-style), supporting
Linear, Polynomial, Gaussian (RBF), and Sigmoid kernels, as well as OvO and OvR
multiclass strategies.

References:
- F. Z. Ruskanda, "Support Vector Machines," IF3170 Inteligensi Artifisial,
  Edunex ITB, 2026.
- J. C. Platt, "Fast Training of Support Vector Machines using Sequential
  Minimal Optimization," in Advances in Kernel Methods - Support Vector
  Learning, MIT Press, 1999.
- R.-E. Fan, P.-H. Chen, and C.-J. Lin, "Working Set Selection Using Second
  Order Information for Training Support Vector Machines," Journal of Machine
  Learning Research (JMLR), vol. 6, pp. 1889-1918, 2005 (LIBSVM WSS2 & kernel
  column caching).
- C.-W. Hsu and C.-J. Lin, "A Comparison of Methods for Multiclass Support
  Vector Machines," IEEE Transactions on Neural Networks, vol. 13, no. 2,
  pp. 415-425, 2002 (One-vs-One and One-vs-Rest strategies).
"""

import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
try:
   from .base import BaseModel
except ImportError:
   from base import BaseModel

_SVM_PALETTE = {
   "blue_main": "#0F4D92",
   "blue_secondary": "#3775BA",
   "red_2": "#E9A6A1",
   "red_strong": "#B64342",
   "highlight": "#FFD700",
   "boundary": "#111827",
   "bounded_sv": "#1F2937",
}


def _svm_style_context():
   return plt.rc_context({
      "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
      "font.size": 14,
      "axes.titlesize": 14,
      "axes.labelsize": 13,
      "xtick.labelsize": 11,
      "ytick.labelsize": 11,
      "axes.spines.right": False,
      "axes.spines.top": False,
      "axes.linewidth": 2.0,
      "legend.frameon": False,
      "svg.fonttype": "none",
   })


def _save_figure(fig, filename, formats=("png",), dpi=300):
   stem, ext = os.path.splitext(str(filename))
   fmt_list = [ext.lstrip(".")] if (ext and formats is None) else list(formats)
   if parent := os.path.dirname(stem):
      os.makedirs(parent, exist_ok=True)
   saved = []
   for fmt in fmt_list:
      clean_fmt = fmt.lstrip(".").lower()
      out_path = f"{stem}.{clean_fmt}"
      fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
      saved.append(out_path)
   return saved

class _KernelColumnCache:
   def __init__(self, max_bytes, col_bytes):
      self.max_cols = max(2, max_bytes // max(col_bytes, 1))
      self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

   def get(self, idx: int):
      col = self._cache.get(idx)

      if col is not None:
         self._cache.move_to_end(idx)
      
      return col

   def put(self, idx: int, col: np.ndarray):
      if idx in self._cache:
         self._cache.move_to_end(idx)
         self._cache[idx] = col
         return
      
      if len(self._cache) >= self.max_cols:
         self._cache.popitem(last=False)
      
      self._cache[idx] = col

class BinarySVC(BaseModel):
   def __init__(
      self,
      C=1.0,
      kernel='linear',
      *,
      degree=3,
      gamma='scale',
      coef0=0.0,
      tol=1e-4,
      solver='smo',
      max_iter=100000,
      cache_size=200.0,
      batch_size=2048,
   ):
      
      self.C = float(C)

      if kernel in ['linear', 'poly', 'polynomial', 'rbf', 'sigmoid']:
         self.kernel = kernel
      else:
         raise ValueError(f"Invalid kernel: {kernel}.")
      
      self.degree = degree
      self.gamma = gamma
      self.coef0 = coef0
      self.tol = tol
      self.solver = solver
      self.max_iter = max_iter
      self.cache_size = cache_size
      self.batch_size = batch_size

      self.classes = None
      self.alpha = None
      self.support_vectors = None
      self.dual_coeffs = None
      self.bias = None
      self._gamma = None
   
   def fit(self, X, y):
      X = np.ascontiguousarray(X, dtype=np.float64)
      y = np.asarray(y)

      if len(np.unique(y)) != 2:
         raise ValueError("BinarySVC is only for binary classification.")

      # Initialize attributes 
      self.classes = np.unique(y)
      y_signed = np.where(y == self.classes[0], -1.0, 1.0)
      self._gamma = self._resolve_gamma(X)

      # Solve QP problem  
      alpha, bias = self._solve_smo(X, y_signed) 

      sv_mask = alpha > min(self.tol, 1e-7, 1e-3 * self.C)
      if not np.any(sv_mask):
         raise RuntimeError(
            f"No support vectors found (max_alpha={alpha.max():.3e}, C={self.C})."
         ) 

      # Save solution 
      self.alpha = alpha[sv_mask]
      self.support_vectors = X[sv_mask] 
      self.dual_coeffs = self.alpha * y_signed[sv_mask] 
      self.bias = float(bias)

      return self
   
   def _resolve_gamma(self, X: np.ndarray) -> float:
      if self.gamma == 'scale':
         var = float(np.var(X))
         return 1.0 / (X.shape[1] * var) if var > 0.0 else 1.0
      
      if self.gamma == 'auto':
         return 1.0 / X.shape[1]
      
      if isinstance(self.gamma, (int, float)):
         return float(self.gamma)
      
      raise ValueError(f"Invalid gamma: {self.gamma}.")
   
   def _solve_smo(self, X: np.ndarray, y: np.ndarray):
      N = X.shape[0]
      C = self.C
      bound_eps = 1e-9 * max(1.0, C)
      step_eps = 1e-12

      alpha = np.zeros(N, dtype=np.float64)
      E = -y.copy()
      diag_K = self._compute_diag(X)
      cache = _KernelColumnCache(int(self.cache_size * 1024 * 1024), N * 8)

      is_rbf = (self.kernel == 'rbf')
      X_norm_sq = np.sum(X ** 2, axis=1) if is_rbf else None
      gamma = self._gamma if self._gamma is not None else 1.0

      def get_col(idx: int) -> np.ndarray:
         col = cache.get(idx)
         if col is None:
            if is_rbf and X_norm_sq is not None:
               dists = X_norm_sq + X_norm_sq[idx] - 2.0 * (X @ X[idx])
               np.maximum(dists, 0.0, out=dists)
               col = np.exp(-gamma * dists)
            else:
               col = self._compute_kernel(X, X[idx : idx + 1]).ravel()
            cache.put(idx, col)
         return col

      pos_y = y == 1.0
      neg_y = ~pos_y

      for _ in range(self.max_iter):
         i_up = (pos_y & (alpha < C - bound_eps)) | (neg_y & (alpha > bound_eps))
         i_low = (neg_y & (alpha < C - bound_eps)) | (pos_y & (alpha > bound_eps))

         if not (np.any(i_up) and np.any(i_low)):
            break

         neg_E = -E
         up_indices = np.flatnonzero(i_up)
         low_indices = np.flatnonzero(i_low)

         i = int(up_indices[np.argmax(neg_E[up_indices])])
         max_neg_Ei = neg_E[i]
         min_neg_Ej = float(np.min(neg_E[low_indices]))

         if max_neg_Ei - min_neg_Ej <= self.tol:
            break

         K_i = get_col(i)

         # Second-Order Working Set Selection (WSS2)
         cand_indices = np.flatnonzero(i_low & (neg_E < max_neg_Ei - step_eps))
         if len(cand_indices) > 0:
            eta_all = np.maximum(
               diag_K[i] + diag_K[cand_indices] - 2.0 * K_i[cand_indices],
               step_eps,
            )
            diff = max_neg_Ei - neg_E[cand_indices]
            j = int(cand_indices[np.argmax((diff * diff) / eta_all)])
         else:
            j = int(low_indices[np.argmin(neg_E[low_indices])])

         K_j = get_col(j)

         y_i, y_j = y[i], y[j]
         a_i_old, a_j_old = alpha[i], alpha[j]

         if y_i != y_j:
            L = max(0.0, a_j_old - a_i_old)
            H = min(C, C + a_j_old - a_i_old)
         else:
            L = max(0.0, a_i_old + a_j_old - C)
            H = min(C, a_i_old + a_j_old)

         if H - L <= step_eps:
            continue

         eta = diag_K[i] + diag_K[j] - 2.0 * K_i[j]
         if eta > step_eps:
            a_j_new = a_j_old + y_j * (E[i] - E[j]) / eta
            a_j_new = float(np.clip(a_j_new, L, H))
         else:
            a_j_new = L if (y_j * (E[i] - E[j]) < 0.0) else H

         if abs(a_j_new - a_j_old) < step_eps * (a_j_new + a_j_old + step_eps):
            continue

         a_i_new = a_i_old + (y_i * y_j) * (a_j_old - a_j_new)

         alpha[i] = a_i_new
         alpha[j] = a_j_new

         delta_i = y_i * (a_i_new - a_i_old)
         delta_j = y_j * (a_j_new - a_j_old)
         E += delta_i * K_i + delta_j * K_j

      sv_tol = 1e-7 * max(1.0, C)
      free_mask = (alpha > sv_tol) & (alpha < C - sv_tol)
      if np.any(free_mask):
         bias = -float(np.mean(E[free_mask]))
      else:
         i_up = (pos_y & (alpha < C - bound_eps)) | (neg_y & (alpha > bound_eps))
         i_low = (neg_y & (alpha < C - bound_eps)) | (pos_y & (alpha > bound_eps))
         neg_E = -E
         b_low = float(np.max(neg_E[i_up])) if np.any(i_up) else 0.0
         b_high = float(np.min(neg_E[i_low])) if np.any(i_low) else 0.0
         bias = 0.5 * (b_low + b_high)

      return np.clip(alpha, 0.0, C), bias

   def _compute_kernel(self, X1, X2):
      if self._gamma is None:
         raise ValueError("Model has not been fitted yet.")

      match self.kernel:

         case 'linear':
            return X1 @ X2.T

         case 'poly' | 'polynomial':
            return (self._gamma * (X1 @ X2.T) + self.coef0) ** self.degree

         case 'rbf':
            x1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            x2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
            norm_sq = x1_sq + x2_sq - 2.0 * (X1 @ X2.T) # ||x - z||² = ||x||² + ||z||² - 2 xᵀ z
            np.maximum(norm_sq, 0.0, out=norm_sq)
            return np.exp(-self._gamma * norm_sq)

         case 'sigmoid':
            return np.tanh(self._gamma * (X1 @ X2.T) + self.coef0)

         case _:
            raise ValueError(f"Invalid kernel: {self.kernel}.")

   def _compute_diag(self, X):
      match self.kernel:

         case 'linear':
            return np.sum(X ** 2, axis=1)
         
         case 'poly' | 'polynomial':
            return (self._gamma * np.sum(X ** 2, axis=1) + self.coef0) ** self.degree
         
         case 'rbf':
            return np.ones(X.shape[0], dtype=np.float64)
         
         case 'sigmoid':
            return np.tanh(self._gamma * np.sum(X ** 2, axis=1) + self.coef0)
         
         case _:
            raise ValueError(f"Invalid kernel: {self.kernel}.")

   def decision_function(self, X):
      if self.support_vectors is None or self.dual_coeffs is None or self.bias is None or self.batch_size is None:
         raise ValueError("Model has not been fitted yet.")

      X = np.asarray(X, dtype=np.float64)
      M = X.shape[0] 
      z = np.empty(M, dtype=np.float64)

      for start in range(0, M, self.batch_size):
         end = min(start + self.batch_size, M)
         K_batch = self._compute_kernel(self.support_vectors, X[start:end])
         z[start:end] = self.dual_coeffs @ K_batch + self.bias

      return z

   def predict(self, X):
      if self.classes is None:
         raise ValueError("Model has not been fitted yet.")

      z = self.decision_function(X)
      return np.where(z >= 0, self.classes[1], self.classes[0])

   def _sv_masks(self):
      if self.alpha is None:
         raise ValueError("Model has not been fitted yet.")
      
      sv_tol = max(1e-7, 1e-5 * self.C)
      free_mask = (self.alpha > sv_tol) & (self.alpha < self.C - sv_tol)
      bounded_mask = ~free_mask
      return free_mask, bounded_mask

   def _legend_handles(self, class_names, distinguish_sv=True):
      handles = [
         mlines.Line2D([], [], linestyle="None", marker="o", markersize=8,
                       markerfacecolor=_SVM_PALETTE["blue_main"], markeredgecolor="white",
                       markeredgewidth=0.8, label=str(class_names[0])),
         mlines.Line2D([], [], linestyle="None", marker="o", markersize=8,
                       markerfacecolor=_SVM_PALETTE["red_strong"], markeredgecolor="white",
                       markeredgewidth=0.8, label=str(class_names[1])),
         mlines.Line2D([], [], color=_SVM_PALETTE["boundary"], linewidth=2.2,
                       label="Decision boundary (f(x) = 0)"),
         mlines.Line2D([], [], color=_SVM_PALETTE["blue_main"], linestyle="--", linewidth=1.6,
                       label="Margin (f(x) = -1)"),
         mlines.Line2D([], [], color=_SVM_PALETTE["red_strong"], linestyle="--", linewidth=1.6,
                       label="Margin (f(x) = +1)"),
      ]
      if distinguish_sv:
         handles.extend([
            mlines.Line2D([], [], linestyle="None", marker="o", markersize=10,
                          markerfacecolor="none", markeredgecolor=_SVM_PALETTE["highlight"],
                          markeredgewidth=2.2, label="Margin SV (0 < alpha < C)"),
            mlines.Line2D([], [], linestyle="None", marker="o", markersize=9,
                          markerfacecolor="none", markeredgecolor=_SVM_PALETTE["bounded_sv"],
                          markeredgewidth=1.2, label="Bounded SV (alpha = C)"),
         ])
      else:
         handles.append(
            mlines.Line2D([], [], linestyle="None", marker="o", markersize=9.5,
                          markerfacecolor="none", markeredgecolor=_SVM_PALETTE["boundary"],
                          markeredgewidth=1.5, label="Support vector")
         )
      return handles

   def visualize_boundary(
      self,
      X,
      y,
      *,
      ax=None,
      feature_names=None,
      class_names=None,
      title=None,
      grid_resolution=250,
      pad_ratio=0.08,
      distinguish_sv=True,
      show_legend=True,
      filename="output/svm_boundary.png",
      save_to_file=False,
      formats=("png",),
      dpi=300,
   ):
      """Plot 2D decision boundary f(x)=0, margins f(x)=±1, and support vectors."""
      if self.classes is None or self.support_vectors is None or self.alpha is None:
         raise ValueError("Model has not been fitted yet.")

      X_arr = np.asarray(X, dtype=np.float64)
      y_arr = np.asarray(y)
      if X_arr.ndim != 2 or X_arr.shape[1] != 2:
         raise ValueError(f"Expected 2D input with 2 features, got shape {X_arr.shape}.")
      if len(X_arr) != len(y_arr):
         raise ValueError("X and y must have the same number of samples.")
      if self.support_vectors.shape[1] != 2:
         raise ValueError(
            f"Model was fitted on {self.support_vectors.shape[1]} features; expected 2."
         )

      feat_x, feat_y = feature_names if feature_names is not None else ("Feature 0", "Feature 1")
      cls_names = (
         tuple(class_names)
         if class_names is not None
         else (f"Class {self.classes[0]}", f"Class {self.classes[1]}")
      )

      x_min, x_max = float(X_arr[:, 0].min()), float(X_arr[:, 0].max())
      y_min, y_max = float(X_arr[:, 1].min()), float(X_arr[:, 1].max())
      x_pad = pad_ratio * (x_max - x_min) if x_max > x_min else 1.0
      y_pad = pad_ratio * (y_max - y_min) if y_max > y_min else 1.0

      xx, yy = np.meshgrid(
         np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution),
         np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution),
      )
      zz = self.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

      own_fig = ax is None
      with _svm_style_context():
         if own_fig:
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
         else:
            fig = ax.figure

         z_low = min(float(zz.min()) - 1.0, -1.0)
         z_high = max(float(zz.max()) + 1.0, 1.0)
         ax.contourf(
            xx,
            yy,
            zz,
            levels=[z_low, 0.0, z_high],
            colors=[_SVM_PALETTE["blue_secondary"], _SVM_PALETTE["red_2"]],
            alpha=0.18,
            extend="both",
         )

         z_min_grid, z_max_grid = float(zz.min()), float(zz.max())
         for level, color, ls, lw in [
            (-1.0, _SVM_PALETTE["blue_main"], "--", 1.6),
            (0.0, _SVM_PALETTE["boundary"], "-", 2.2),
            (1.0, _SVM_PALETTE["red_strong"], "--", 1.6),
         ]:
            if z_min_grid < level < z_max_grid:
               ax.contour(xx, yy, zz, levels=[level], colors=[color], linestyles=[ls], linewidths=[lw])

         mask_neg = y_arr == self.classes[0]
         mask_pos = y_arr == self.classes[1]
         ax.scatter(
            X_arr[mask_neg, 0],
            X_arr[mask_neg, 1],
            c=_SVM_PALETTE["blue_main"],
            s=36,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
         )
         ax.scatter(
            X_arr[mask_pos, 0],
            X_arr[mask_pos, 1],
            c=_SVM_PALETTE["red_strong"],
            s=36,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
         )

         free_mask, bounded_mask = self._sv_masks()
         if distinguish_sv:
            if np.any(bounded_mask):
               ax.scatter(
                  self.support_vectors[bounded_mask, 0],
                  self.support_vectors[bounded_mask, 1],
                  s=80,
                  facecolors="none",
                  edgecolors=_SVM_PALETTE["bounded_sv"],
                  linewidths=1.0,
                  alpha=0.75,
                  zorder=4,
               )
            if np.any(free_mask):
               ax.scatter(
                  self.support_vectors[free_mask, 0],
                  self.support_vectors[free_mask, 1],
                  s=155,
                  facecolors="none",
                  edgecolors=_SVM_PALETTE["boundary"],
                  linewidths=2.4,
                  zorder=5,
               )
               ax.scatter(
                  self.support_vectors[free_mask, 0],
                  self.support_vectors[free_mask, 1],
                  s=130,
                  facecolors="none",
                  edgecolors=_SVM_PALETTE["highlight"],
                  linewidths=1.8,
                  zorder=6,
               )
         else:
            ax.scatter(
               self.support_vectors[:, 0],
               self.support_vectors[:, 1],
               s=100,
               facecolors="none",
               edgecolors=_SVM_PALETTE["boundary"],
               linewidths=1.3,
               zorder=4,
            )

         ax.set_xlabel(feat_x)
         ax.set_ylabel(feat_y)
         ax.set_xlim(x_min - x_pad, x_max + x_pad)
         ax.set_ylim(y_min - y_pad, y_max + y_pad)
         ax.spines["top"].set_visible(False)
         ax.spines["right"].set_visible(False)

         if title is None:
            title = f"SVM ({self.kernel}) | SV: {len(self.support_vectors)} ({int(free_mask.sum())} margin)"
         ax.set_title(title, pad=10)

         if show_legend:
            ax.legend(
               handles=self._legend_handles(cls_names, distinguish_sv=distinguish_sv),
               loc="best",
               frameon=False,
               fontsize=10,
            )

         if own_fig:
            fig.tight_layout(pad=1.0)
            if save_to_file:
               _save_figure(fig, filename, formats=formats, dpi=dpi)
            plt.close(fig)
            return fig

      return ax

class SVC(BaseModel):
   def __init__(
      self,
      C=1.0,
      kernel='linear',
      multiclass='ovo',
      *,
      degree=3,
      gamma='scale',
      coef0=0.0,
      tol=1e-6,
      solver='smo',
      max_iter=100000,
      cache_size=200.0,
      batch_size=2048,
   ):
      
      self.C = C

      if kernel in ['linear', 'poly', 'polynomial', 'rbf', 'sigmoid']:
         self.kernel = kernel
      else:
         raise ValueError(f"Invalid kernel: {kernel}.")

      if multiclass in ['ovo', 'ovr']:
         self.multiclass = multiclass
      else:
         raise ValueError(f"Invalid multiclass: {multiclass}")
      
      self.degree = degree
      self.gamma = gamma
      self.coef0 = coef0
      self.tol = tol
      self.solver = solver
      self.max_iter = max_iter
      self.cache_size = cache_size
      self.batch_size = batch_size

      self.classes = None
      self.estimators = None

   def _make_binary_svc(self):
      return BinarySVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            solver=self.solver,
            max_iter=self.max_iter,
            cache_size=self.cache_size,
            batch_size=self.batch_size,
      )

   def fit(self, X, y):
      X = np.asarray(X, dtype=np.float64)
      y = np.asarray(y)

      self.classes = np.unique(y)
      K = len(self.classes)

      if K < 2:
         raise ValueError(f"Target contains only {K} class.")

      # Binary SVC
      if K == 2: 
         svm = self._make_binary_svc()
         svm.fit(X, y)
         self.estimators = [(0, 1, svm)]
         return self

      # Multiclass SVC
      estimators = []
      match self.multiclass:

         case 'ovo':
            for a in range(K):
               for b in range(a + 1, K):
                  pair_mask = (y == self.classes[a]) | (y == self.classes[b])
                  svm_ab = self._make_binary_svc()
                  svm_ab.fit(X[pair_mask], y[pair_mask])
                  estimators.append((a, b, svm_ab))

         case 'ovr':
            for k in range(K):
               y_k = np.where(y == self.classes[k], 1, 0)
               svm_k = self._make_binary_svc()
               svm_k.fit(X, y_k)
               estimators.append((-k, k, svm_k))

         case _:
            raise ValueError(f"Invalid multiclass: {self.multiclass}")

      self.estimators = estimators
      return self

   def decision_function(self, X):
      if self.classes is None or self.estimators is None:
         raise ValueError("Model has not been fitted yet.")

      X = np.asarray(X, dtype=np.float64)
      M = X.shape[0]
      K = len(self.classes)

      # Binary SVC
      if K == 2: 
         return self.estimators[0][2].decision_function(X)

      # Multiclass SVC
      match self.multiclass:

         case 'ovo':
            votes = np.zeros((M, K)) 
            margins = np.zeros((M, K))

            for a, b, svm in self.estimators: 
               f_ab = svm.decision_function(X)

               votes[f_ab < 0, a] += 1
               votes[f_ab >= 0, b] += 1
               
               margins[:, a] -= f_ab
               margins[:, b] += f_ab

            transformed_margins = margins / (2.0 * (np.abs(margins) + 1.0)) 
            return votes + transformed_margins
            # votes: number of wins  
            # transformed_margins: confidence transformed to (-0.5,0.5) for tiebreaker

         case 'ovr':
            return np.column_stack([svm.decision_function(X) for _, _, svm in self.estimators])

         case _:
            raise ValueError(f"Invalid multiclass: {self.multiclass}")

   def predict(self, X):
      if self.classes is None or self.estimators is None:
         raise ValueError("Model has not been fitted yet.")
      
      z = self.decision_function(X)
      K = len(self.classes)

      if K == 2: 
         return np.where(z >= 0, self.classes[1], self.classes[0])
      
      else: 
         return self.classes[np.argmax(z, axis=1)]

   def visualize_boundary(self, X, y, **kwargs):
      if self.classes is None or self.estimators is None:
         raise ValueError("Model has not been fitted yet.")
      if len(self.classes) != 2:
         raise ValueError("visualize_boundary only supports binary classification.")
      return self.estimators[0][2].visualize_boundary(X, y, **kwargs)


def visualize_svm_kernels(
   models,
   X,
   y,
   *,
   feature_names=None,
   class_names=("Not Canceled (0)", "Canceled (1)"),
   grid_resolution=250,
   pad_ratio=0.08,
   distinguish_sv=True,
   filename="output/svm_kernels_2d_boundary.png",
   save_to_file=True,
   formats=("png",),
   dpi=300,
):
   items = list(models.items()) if isinstance(models, dict) else list(models)
   if not items:
      raise ValueError("models must contain at least one fitted SVM model.")

   X_arr = np.asarray(X, dtype=np.float64)
   y_arr = np.asarray(y)

   with _svm_style_context():
      fig, axes = plt.subplots(
         1,
         len(items),
         figsize=(6.0 * len(items), 5.8),
         sharex=True,
         sharey=True,
      )
      axes_arr = np.atleast_1d(axes).flatten()

      legend_handles = None
      for idx, ((label, model), ax) in enumerate(zip(items, axes_arr)):
         if isinstance(model, SVC):
            if model.estimators is None or len(model.estimators) == 0:
                  raise ValueError(f"SVC model '{label}' has not been fitted yet.")
            est = model.estimators[0][2]
         else:
            est = model
         if not isinstance(est, BinarySVC) or est.classes is None or est.support_vectors is None:
            raise ValueError(f"Model '{label}' must be a fitted binary SVC/BinarySVC.")

         acc = float(np.mean(model.predict(X_arr) == y_arr))
         free_mask, _ = est._sv_masks()
         panel_title = (
            f"{label}\n"
            f"Acc: {acc:.1%} | SV: {len(est.support_vectors)} ({int(free_mask.sum())} margin)"
         )
         est.visualize_boundary(
            X_arr,
            y_arr,
            ax=ax,
            feature_names=feature_names,
            class_names=class_names,
            title=panel_title,
            grid_resolution=grid_resolution,
            pad_ratio=pad_ratio,
            distinguish_sv=distinguish_sv,
            show_legend=False,
            save_to_file=False,
         )
         if idx > 0:
            ax.set_ylabel("")
         if legend_handles is None:
            legend_handles = est._legend_handles(class_names, distinguish_sv=distinguish_sv)

      if legend_handles:
         fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=11.5,
            columnspacing=1.6,
            handletextpad=0.6,
         )

      fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
      if save_to_file:
         _save_figure(fig, filename, formats=formats, dpi=dpi)
      plt.close(fig)
      return fig
