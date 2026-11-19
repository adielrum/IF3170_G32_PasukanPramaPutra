import numpy as np
import scipy.sparse as sparse
import clarabel
try:
   from .base import BaseModel
except ImportError:
   from base import BaseModel
   
# Algoritma SMO: https://cs229.stanford.edu/materials/smo.pdf
class BinarySVC(BaseModel):
   def __init__(self, C=1.0, kernel='linear', *, degree=3, gamma="scale", coef0=0.0, tol=1e-6):
      
      self.C = C

      if kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
         self.kernel = kernel
      else:
         raise ValueError(f"Invalid kernel: {kernel}.")
      
      self.degree = degree
      self.gamma = gamma
      self.coef0 = coef0
      self.tol = tol

      self.alpha = None
      self.support_vectors = None
      self.dual_coeffs = None
      self.bias = None
      self._gamma = None
   
   def fit(self, X, y):
      X = np.asarray(X, dtype=np.float64)
      y = np.asarray(y)

      if len(np.unique(y)) != 2:
         raise ValueError("BinarySVC is only for binary classification.")

      self.classes = np.unique(y)
      y_signed = np.where(y == self.classes[0], -1, 1)
      
      if self.gamma == 'scale':
         self._gamma = 1.0 / (X.shape[1] * np.var(X)) if np.var(X) > 0 else 1.0
      
      elif self.gamma == 'auto':
         self._gamma = 1.0 / X.shape[1]
      
      elif isinstance(self.gamma, (int, float)):
         self._gamma = float(self.gamma)
      
      else:
         raise ValueError(f"Invalid gamma: {self.gamma}.")

      K = self._compute_kernel(X, X)
      alpha = self._solve_qp(K, y_signed)

      sv_mask = alpha > self.tol 
      if not np.any(sv_mask): 
         raise RuntimeError("No support vectors found.")

      alpha_sv = alpha[sv_mask]
      y_sv = y_signed[sv_mask]
      x_sv = X[sv_mask]
      K_sv = K[np.ix_(sv_mask, sv_mask)]
      
      self.alpha = alpha_sv
      self.support_vectors = x_sv
      self.dual_coeffs = alpha_sv * y_sv

      sum_sv = self.dual_coeffs @ K_sv # Σ αᵢ yᵢ K(xᵢ, xⱼ)
      free_sv_mask = alpha_sv < self.C - self.tol

      if np.any(free_sv_mask):
         self.bias = float(np.mean(y_sv[free_sv_mask] - sum_sv[free_sv_mask])) # free support vectors  
      else:
         self.bias = float(np.mean(y_sv - sum_sv)) # all support vectors

      return self
   
   def _solve_qp(self, K, y):
      N = K.shape[0]
      y = np.asarray(y, dtype=np.float64)

      # Objective: # 1/2 αᵀ P α + qᵀ α 
      Q = K * np.outer(y, y)
      Q = 0.5 * (Q + Q.T)
      P = sparse.triu(Q, format="csc")
      q = -np.ones(N, dtype=np.float64)

      # Constraints: Aα + s = b
      A = sparse.vstack(
         [
            sparse.csc_matrix(y.reshape(1, -1)),  # yᵀα + s₁ = 0
            -sparse.eye(N, format="csc"), # -α + s₂ = 0 
            sparse.eye(N, format="csc"), # α + s₃ = C 
         ],
         format="csc",
      )
      b = np.concatenate(
         [[0.0], 
         np.zeros(N), 
         np.full(N, self.C)]
      )
      cones = [
         clarabel.ZeroConeT(1), # s₁ = 0 
         clarabel.NonnegativeConeT(2 * N) # s₂, s₃ ≥ 0
      ]
      
      settings = clarabel.DefaultSettings()
      settings.verbose = False
      tol = min(self.tol * 1e-2, 1e-8)
      settings.tol_gap_abs = tol
      settings.tol_gap_rel = tol
      settings.tol_feas = tol

      # Solve QP 
      solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
      solution = solver.solve()

      if solution.status != clarabel.SolverStatus.Solved:
         raise RuntimeError(f"QP solver failed: {solution.status}")

      alpha = np.asarray(solution.x, dtype=np.float64)
      return np.clip(alpha, 0.0, self.C)

   def _compute_kernel(self, X1, X2):
      if self._gamma is None:
         raise ValueError("Model has not been fitted yet.")

      match self.kernel:

         case 'linear':
            return X1 @ X2.T

         case 'poly':
            return (self._gamma * (X1 @ X2.T) + self.coef0) ** self.degree

         case 'rbf':
            norm_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2.0 * (X1 @ X2.T) # ||x - z||² = ||x||² + ||z||² - 2 xᵀ z
            return np.exp(-self._gamma * np.maximum(norm_sq, 0.0))

         case 'sigmoid':
            return np.tanh(self._gamma * (X1 @ X2.T) + self.coef0)

         case _:
            raise ValueError(f"Invalid kernel: {self.kernel}.")

   def decision_function(self, X):
      if self.alpha is None or self.support_vectors is None or self.dual_coeffs is None or self.bias is None:
         raise ValueError("Model has not been fitted yet.")

      X = np.asarray(X, dtype=np.float64)
      K = self._compute_kernel(self.support_vectors, X)
      return self.dual_coeffs @ K + self.bias

   def predict(self, X):
      if self.alpha is None or self.support_vectors is None or self.dual_coeffs is None or self.bias is None:
         raise ValueError("Model has not been fitted yet.")

      z = self.decision_function(X)
      return np.where(z >= 0, self.classes[1], self.classes[0])

class SVC(BaseModel):

   def __init__(self, C=1.0, kernel='linear', multiclass='ovo', *, degree=3, gamma="scale", coef0=0.0, tol=1e-6):
      
      self.C = C

      if kernel in ['linear', 'polynomial', 'rbf', 'sigmoid']:
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
