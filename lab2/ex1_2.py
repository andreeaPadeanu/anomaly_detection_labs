import numpy as np
import matplotlib.pyplot as plt

np.random.seed(15)
a1, a2, c = 1.5, -0.8, 0.5
means = [0, 1]
variances = [0.1, 1]

for idx, (mean, var) in enumerate(zip(means, variances)):

    x1_reg = np.random.normal(mean, var, 100)
    x2_reg = np.random.normal(mean, var, 100)
    y_reg = a1 * x1_reg + a2 * x2_reg + c + np.random.normal(mean, np.sqrt(var), 100)

    x1_hx = np.random.normal(mean, var * 3, 100)
    x2_hx = np.random.normal(mean, var * 3, 100)
    y_hx = a1 * x1_hx + a2 * x2_hx + c + np.random.normal(mean, np.sqrt(var), 100)

    x1_hy = np.random.normal(mean, var, 100)
    x2_hy = np.random.normal(mean, var, 100)
    y_hy = a1 * x1_hy + a2 * x2_hy + c + np.random.normal(mean, np.sqrt(var * 5), 100)

    x1_both = np.random.normal(mean, var * 3, 100)
    x2_both = np.random.normal(mean, var * 3, 100)
    y_both = a1 * x1_both + a2 * x2_both + c + np.random.normal(mean, np.sqrt(var * 5), 100)

    x1 = np.concatenate([x1_reg, x1_hx, x1_hy, x1_both])
    x2 = np.concatenate([x2_reg, x2_hx, x2_hy, x2_both])
    y = np.concatenate([y_reg, y_hx, y_hy, y_both])

    X = np.column_stack((np.ones_like(x1), x1, x2))
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    H = U @ U.T
    leverage = np.diag(H)

    top_points = np.argsort(leverage)[-20:]

    plt.figure(figsize=(6, 5))
    plt.scatter(x1, x2, c=leverage, cmap='viridis', alpha=0.6, label='All points')
    plt.scatter(x1[top_points], x2[top_points], color='red', label='High leverage', s=60)
    plt.colorbar(label='Leverage value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'2D Model: μ={mean}, σ²={var}')
    plt.legend()
    plt.tight_layout()

    plt.savefig("ex1_2.svg")
