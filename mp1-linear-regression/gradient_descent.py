import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    x = df["YearsExperience"].to_numpy()
    y = df["Salary"].to_numpy()
    return x, y


def compute_cost(x, y, w, b):
    m = len(x)
    errors = (w * x + b) - y
    return np.sum(errors ** 2) / (2 * m)


def compute_gradient(x, y, w, b):
    """Partial derivatives of J with respect to w and b."""
    m = len(x)
    errors = (w * x + b) - y
    dj_dw = np.sum(errors * x) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    """Returns final w, b and the cost recorded at every iteration."""
    w, b = w_init, b_init
    cost_history = []

    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # simultaneous update: both gradients are computed before either parameter changes
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost_history.append(compute_cost(x, y, w, b))

    return w, b, cost_history


def numerical_gradient(x, y, w, b, eps=1e-4):
    """Finite-difference estimate of the gradient, used only to check compute_gradient."""
    dw = (compute_cost(x, y, w + eps, b) - compute_cost(x, y, w - eps, b)) / (2 * eps)
    db = (compute_cost(x, y, w, b + eps) - compute_cost(x, y, w, b - eps)) / (2 * eps)
    return dw, db


if __name__ == "__main__":
    x, y = load_data(BASE_DIR / "data" / "Salary_Data.csv")

    # Sanity check 1: analytic gradient matches the numerical one
    g_analytic = compute_gradient(x, y, 5000.0, 20000.0)
    g_numeric = numerical_gradient(x, y, 5000.0, 20000.0)
    assert np.allclose(g_analytic, g_numeric, rtol=1e-4), "Gradient mismatch"
    print("Gradient check passed")

    # Main run
    alpha = 0.01
    num_iters = 20000
    w, b, history = gradient_descent(x, y, 0.0, 0.0, alpha, num_iters)
    print(f"alpha={alpha}, iterations={num_iters}")
    print(f"w = {w:.2f}, b = {b:.2f}, final cost = {history[-1]:.2f}")

    # Sanity check 2: compare against NumPy's closed-form least squares fit
    w_ref, b_ref = np.polyfit(x, y, 1)
    print(f"np.polyfit reference: w = {w_ref:.2f}, b = {b_ref:.2f}")

    # Plot 1: loss curve
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w, b)")
    plt.title(f"Loss curve (alpha = {alpha})")
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # Plot 2: fitted line over the data
    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, w * x + b, color="red", label=f"Fit: y = {w:.0f}x + {b:.0f}")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Linear regression fit from gradient descent")
    plt.legend()
    plt.savefig(PLOTS_DIR / "fitted_line.png", dpi=150)
    plt.close()

    # Plot 3: loss curves for different learning rates
    alphas = [0.001, 0.01, 0.03]
    plt.figure()
    for a in alphas:
        _, _, h = gradient_descent(x, y, 0.0, 0.0, a, 5000)
        plt.plot(h, label=f"alpha = {a}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w, b)")
    plt.yscale("log")
    plt.title("Effect of learning rate on convergence")
    plt.legend()
    plt.savefig(PLOTS_DIR / "learning_rates.png", dpi=150)
    plt.close()

    print("Plots saved to", PLOTS_DIR)