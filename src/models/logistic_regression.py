"""
Multiclass / Binary Logistic (Softmax) Regression with L1, L2, and Elastic Net
regularization trained via mini-batch gradient descent, plus log-loss contour
trajectory animation.

References:
- F. Z. Ruskanda, "Linear Regression & Logistic Regression," IF3170 Inteligensi
  Artifisial, Edunex ITB, 2026.
- M. L. Khodra, "Logistic Regression: What & Why" & "Stochastic Gradient
  Ascent," IF3270 / YouTube, 2021.
- J. Mody, "Numerically Stable Softmax," https://jaykmody.com/blog/stable-softmax/
- K. Sankar, "Gradients for multi class classification with Softmax," DSP LOG,
  2025, https://dsplog.com/2025/06/22/gradients-for-multi-class-classification-with-softmax/
- M. Brenndoerfer, "Elastic Net Regularization: Math & Python," Machine Learning
  from Scratch, 2025, https://mbrenndoerfer.com/writing/elastic-net-regularization-complete-guide-mathematical-foundations-python-implementation
- D. P. Kingma & J. Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015,
  https://arxiv.org/abs/1412.6980
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

class LogisticRegression(BaseModel):
    def __init__(
        self,
        learning_rate=0.01,
        num_iterations=1000,
        batch_size=None,
        *,
        l1_lambda=0.0,
        l2_lambda=0.0,
        optimizer="sgd",
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        random_state=None,
    ):
        if optimizer not in ("sgd", "adam"):
            raise ValueError(f"Invalid optimizer: {optimizer}. Choose 'sgd' or 'adam'.")

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size

        self.weights = None
        self.bias = None
        self.history = None

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.random_state = random_state

    def _softmax(self, z):
        # Stable softmax from https://jaykmody.com/blog/stable-softmax/ 
        c = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - c) 
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_log_loss(self, X, y_onehot):
        probs = np.clip(self.predict_proba(X), 1e-15, 1.0)
        ce_loss = -np.mean(np.sum(y_onehot * np.log(probs), axis=1))
        reg_loss = (
            self.l1_lambda * np.sum(np.abs(self.weights))
            + 0.5 * self.l2_lambda * np.sum(self.weights ** 2)
        )
        return float(ce_loss + reg_loss)

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        
        y = self.predict_proba(X)
        return np.argmax(y, axis=1)

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        return self._softmax(z)

    def fit(self, X, y, record_history=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        rng = np.random.default_rng(self.random_state)

        if self.batch_size is None:
            batch_size = num_samples
        elif self.batch_size >= num_samples:
            batch_size = num_samples
        elif self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")
        else:
            batch_size = self.batch_size

        # One-hot encoding
        y_reshaped = np.zeros((num_samples, num_classes), dtype=np.float64)
        y_reshaped[np.arange(num_samples), y] = 1.0

        self.weights = np.zeros((num_features, num_classes), dtype=np.float64)
        self.bias = np.zeros(num_classes, dtype=np.float64)

        # Adam optimizer state (Kingma & Ba, ICLR 2015)
        if self.optimizer == "adam":
            m_w = np.zeros_like(self.weights)
            v_w = np.zeros_like(self.weights)
            m_b = np.zeros_like(self.bias)
            v_b = np.zeros_like(self.bias)
            t = 0

        if record_history:
            self.history = {
                "weights": [self.weights.copy()],
                "bias": [self.bias.copy()],
                "loss": [self._compute_log_loss(X, y_reshaped)],
            }
        else:
            self.history = None

        for _ in range(self.num_iterations):
            indices = rng.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_reshaped[indices]

            # Mini-Batch Gradient Descent / Adam
            for j in range(0, num_samples, batch_size):
                X_batch = X_shuffled[j:j+batch_size]
                y_batch = y_shuffled[j:j+batch_size]
                current_batch_size = len(X_batch)

                y_predicted = self.predict_proba(X_batch)

                # Gradients: https://dsplog.com/2025/06/22/gradients-for-multi-class-classification-with-softmax/#gsc.tab=0 
                dw = (1 / current_batch_size) * (X_batch.T @ (y_predicted - y_batch)) # 1/N * Xᵀ * (ŷ - y)
                db = (1 / current_batch_size) * np.sum(y_predicted - y_batch, axis=0) # 1/N * 1ᵀ * (ŷ - y)

                # Regularization : https://mbrenndoerfer.com/writing/elastic-net-regularization-complete-guide-mathematical-foundations-python-implementation
                dl1 = self.l1_lambda * np.sign(self.weights) # L1  
                dl2 = self.l2_lambda * self.weights # L2  

                grad_w = dw + dl1 + dl2
                grad_b = db

                if self.optimizer == "adam":
                    t += 1
                    m_w = self.beta1 * m_w + (1.0 - self.beta1) * grad_w
                    v_w = self.beta2 * v_w + (1.0 - self.beta2) * (grad_w ** 2)
                    m_b = self.beta1 * m_b + (1.0 - self.beta1) * grad_b
                    v_b = self.beta2 * v_b + (1.0 - self.beta2) * (grad_b ** 2)

                    m_w_hat = m_w / (1.0 - self.beta1 ** t)
                    v_w_hat = v_w / (1.0 - self.beta2 ** t)
                    m_b_hat = m_b / (1.0 - self.beta1 ** t)
                    v_b_hat = v_b / (1.0 - self.beta2 ** t)

                    self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
                    self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
                else:
                    self.weights -= self.learning_rate * grad_w
                    self.bias -= self.learning_rate * grad_b

            if record_history:
                self.history["weights"].append(self.weights.copy())
                self.history["bias"].append(self.bias.copy())
                self.history["loss"].append(self._compute_log_loss(X, y_reshaped))

        return self

    def animate_loss_contour(
        self,
        X,
        y,
        filename="logistic_regression_contour.gif",
        save_to_file=True,
        snapshot_filename=None,
        param_mode="intercept_weight",
        feature_indices=(0, 1),
        grid_size=80,
        n_frames=75,
        fps=15,
        dpi=150,
        param_names=None,
    ):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        if self.history is None or len(self.history["loss"]) < 2:
            raise ValueError("Training history not found. Call fit(X, y, record_history=True) first.")
        if self.weights.shape[1] != 2:
            raise ValueError("animate_loss_contour is only supported for binary classification.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        w_hist = np.asarray(self.history["weights"], dtype=np.float64)  # (T, D, 2)
        b_hist = np.asarray(self.history["bias"], dtype=np.float64)     # (T, 2)
        losses = np.asarray(self.history["loss"], dtype=np.float64)     # (T,)

        # Extract effective binary parameter trajectory (theta_0, theta_1)
        if param_mode == "intercept_weight":
            f0 = feature_indices[0] if isinstance(feature_indices, (tuple, list)) else int(feature_indices)
            theta0_hist = b_hist[:, 1] - b_hist[:, 0]
            theta1_hist = w_hist[:, f0, 1] - w_hist[:, f0, 0]
            default_names = (r"$\theta_0$ (Bias)", rf"$\theta_1$ (Feature {f0})")

            # Fixed contribution from any other features (zero when D == 1)
            w_diff_final = self.weights[:, 1] - self.weights[:, 0]
            other_mask = np.ones(X.shape[1], dtype=bool)
            other_mask[f0] = False
            base_logits = X[:, other_mask] @ w_diff_final[other_mask] if np.any(other_mask) else np.zeros(len(X))
            other_reg = (
                self.l1_lambda * np.sum(np.abs(self.weights[other_mask]))
                + 0.5 * self.l2_lambda * np.sum(self.weights[other_mask] ** 2)
            )
            x1 = X[:, f0]
        elif param_mode == "two_weights":
            f0, f1 = feature_indices
            theta0_hist = w_hist[:, f0, 1] - w_hist[:, f0, 0]
            theta1_hist = w_hist[:, f1, 1] - w_hist[:, f1, 0]
            default_names = (rf"$\theta_0$ (Feature {f0})", rf"$\theta_1$ (Feature {f1})")

            w_diff_final = self.weights[:, 1] - self.weights[:, 0]
            b_diff_final = float(self.bias[1] - self.bias[0])
            other_mask = np.ones(X.shape[1], dtype=bool)
            other_mask[[f0, f1]] = False
            base_logits = np.full(len(X), b_diff_final, dtype=np.float64)
            if np.any(other_mask):
                base_logits = base_logits + X[:, other_mask] @ w_diff_final[other_mask]
            other_reg = (
                self.l1_lambda * np.sum(np.abs(self.weights[other_mask]))
                + 0.5 * self.l2_lambda * np.sum(self.weights[other_mask] ** 2)
            )
            x0, x1 = X[:, f0], X[:, f1]
        else:
            raise ValueError(f"Invalid param_mode: {param_mode}. Use 'intercept_weight' or 'two_weights'.")

        xlabel, ylabel = param_names if param_names is not None else default_names

        # Compute contour grid bounds with generous margin around trajectory
        t0_min, t0_max = float(theta0_hist.min()), float(theta0_hist.max())
        t1_min, t1_max = float(theta1_hist.min()), float(theta1_hist.max())
        t0_span = max(t0_max - t0_min, 0.5)
        t1_span = max(t1_max - t1_min, 0.5)
        pad = 0.35

        t0_grid = np.linspace(t0_min - pad * t0_span, t0_max + pad * t0_span, grid_size)
        t1_grid = np.linspace(t1_min - pad * t1_span, t1_max + pad * t1_span, grid_size)
        T0, T1 = np.meshgrid(t0_grid, t1_grid)
        Z = np.empty_like(T0, dtype=np.float64)

        # Row-wise (memory-safe) exact log-loss evaluation on the contour grid
        for r in range(grid_size):
            t0_row = T0[r, :, None]  # (grid_size, 1)
            t1_row = T1[r, :, None]  # (grid_size, 1)
            if param_mode == "intercept_weight":
                logits = base_logits[None, :] + t0_row + t1_row * x1[None, :]
                reg = self.l1_lambda * np.abs(T1[r]) + 0.25 * self.l2_lambda * (T1[r] ** 2) + other_reg
            else:
                logits = base_logits[None, :] + t0_row * x0[None, :] + t1_row * x1[None, :]
                reg = (
                    self.l1_lambda * (np.abs(T0[r]) + np.abs(T1[r]))
                    + 0.25 * self.l2_lambda * (T0[r] ** 2 + T1[r] ** 2)
                    + other_reg
                )
            # Numerically stable binary log-loss: mean(log(1 + exp(z)) - y * z)
            Z[r] = np.mean(np.logaddexp(0.0, logits) - y[None, :] * logits, axis=1) + reg

        # Sample frames smoothly: denser early during steep descent, uniform toward convergence
        total_steps = len(losses)
        if total_steps <= n_frames:
            frame_indices = np.arange(total_steps)
        else:
            u = np.linspace(0.0, 1.0, n_frames)
            curved = np.round((u ** 1.6) * (total_steps - 1)).astype(int)
            frame_indices = np.unique(np.concatenate(([0], curved, [total_steps - 1])))

        # Publication-ready house style (figures4papers semantic palette)
        palette = {
            "blue_main": "#0F4D92",
            "blue_secondary": "#3775BA",
            "red_strong": "#B64342",
            "teal": "#42949E",
            "neutral": "#CFCECE",
            "highlight": "#FFD700",
        }
        contour_cmap = LinearSegmentedColormap.from_list(
            "f4p_blue_teal", ["#F4F8FB", "#D2E4F1", "#8AB8D7", "#3775BA", "#0F4D92"]
        )

        style_ctx = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.linewidth": 2.0,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }

        with plt.rc_context(style_ctx):
            fig, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
            fig.patch.set_facecolor("white")

            # Left panel: Log-loss contour surface (denser levels near the minimum)
            z_min, z_max = float(Z.min()), float(Z.max())
            z_span = max(z_max - z_min, 1e-5)
            levels = np.concatenate(([z_min], z_min + np.geomspace(z_span * 1e-3, z_span, 20)))
            cf = ax_c.contourf(T0, T1, Z, levels=levels, cmap=contour_cmap, alpha=0.88)
            cs = ax_c.contour(T0, T1, Z, levels=levels[2::2], colors="#1B365D", linewidths=0.75, alpha=0.55)
            ax_c.clabel(cs, inline=True, fontsize=8.5, fmt="%.3f")
            cbar = fig.colorbar(cf, ax=ax_c, pad=0.02)
            cbar.set_label("Log-Loss", fontsize=12)
            cbar.outline.set_linewidth(1.2)

            ax_c.scatter(
                [theta0_hist[0]], [theta1_hist[0]],
                color="#42949E", edgecolor="black", s=85, zorder=5, label=r"Start $\theta^{(0)}$"
            )
            ax_c.scatter(
                [theta0_hist[-1]], [theta1_hist[-1]],
                color=palette["highlight"], edgecolor="black", marker="*", s=180, zorder=5, label=r"Final $\theta^{(T)}$"
            )
            (traj_line,) = ax_c.plot([], [], color=palette["red_strong"], lw=2.4, zorder=4, label="Trajectory")
            (curr_dot,) = ax_c.plot(
                [], [], marker="o", markersize=8,
                color=palette["red_strong"], markeredgecolor="black", markeredgewidth=1.2, zorder=6
            )

            ax_c.set_xlabel(xlabel)
            ax_c.set_ylabel(ylabel)
            ax_c.set_title("Log-Loss Contour & Parameter Trajectory", fontweight="bold", pad=10)
            ax_c.spines["top"].set_visible(False)
            ax_c.spines["right"].set_visible(False)
            ax_c.legend(frameon=False, loc="best", fontsize=10.5)

            # Right panel: Log-loss vs iteration
            iters = np.arange(total_steps)
            ax_l.plot(iters, losses, color=palette["neutral"], lw=1.8, linestyle="--", alpha=0.8, label="Full curve")
            (loss_line,) = ax_l.plot([], [], color=palette["blue_main"], lw=2.6, label="Training loss")
            (loss_dot,) = ax_l.plot(
                [], [], marker="o", markersize=8,
                color=palette["red_strong"], markeredgecolor="black", markeredgewidth=1.2, zorder=5
            )

            loss_span = max(float(losses.max() - losses.min()), 1e-4)
            ax_l.set_xlim(0, max(1, total_steps - 1))
            ax_l.set_ylim(float(losses.min()) - 0.08 * loss_span, float(losses.max()) + 0.08 * loss_span)
            ax_l.set_xlabel("Iteration")
            ax_l.set_ylabel("Log-Loss")
            ax_l.set_title("Log-Loss Convergence", fontweight="bold", pad=10)
            ax_l.spines["top"].set_visible(False)
            ax_l.spines["right"].set_visible(False)
            ax_l.legend(frameon=False, loc="upper right", fontsize=10.5)

            status_text = fig.suptitle("", fontsize=12.5, fontweight="bold", y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.94], pad=1.4)

            def update(frame_pos):
                idx = int(frame_indices[frame_pos])
                traj_line.set_data(theta0_hist[: idx + 1], theta1_hist[: idx + 1])
                curr_dot.set_data([theta0_hist[idx]], [theta1_hist[idx]])
                loss_line.set_data(iters[: idx + 1], losses[: idx + 1])
                loss_dot.set_data([iters[idx]], [losses[idx]])
                status_text.set_text(
                    rf"Iteration {idx}/{total_steps - 1}   |   "
                    rf"$\theta_0$ = {theta0_hist[idx]:.4f}, $\theta_1$ = {theta1_hist[idx]:.4f}   |   "
                    rf"Log-Loss = {losses[idx]:.4f}"
                )
                return traj_line, curr_dot, loss_line, loss_dot, status_text

            anim = FuncAnimation(
                fig,
                update,
                frames=len(frame_indices),
                interval=int(1000 / fps),
                blit=False,
            )

            if save_to_file:
                ext = os.path.splitext(filename)[1].lower()
                if not ext:
                    filename += ".gif"
                    ext = ".gif"
                if parent := os.path.dirname(filename):
                    os.makedirs(parent, exist_ok=True)

                if ext == ".mp4":
                    writer = FFMpegWriter(fps=fps, bitrate=1800)
                else:
                    writer = PillowWriter(fps=fps)
                anim.save(filename, writer=writer, dpi=dpi)

            if snapshot_filename:
                if parent := os.path.dirname(snapshot_filename):
                    os.makedirs(parent, exist_ok=True)
                update(len(frame_indices) - 1)
                fig.savefig(snapshot_filename, dpi=300, bbox_inches="tight")

            plt.close(fig)

        return anim
