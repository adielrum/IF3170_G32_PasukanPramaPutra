import numpy as np
from collections import OrderedDict
try:
   from .base import BaseModel
except ImportError:
   from base import BaseModel

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

      def get_col(idx: int) -> np.ndarray:
         col = cache.get(idx)
         if col is None:
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
