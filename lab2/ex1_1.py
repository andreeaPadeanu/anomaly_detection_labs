import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
a, b = 1, 1
means = [0, 1]
variances = [0.1, 1]

for idx, (mean, var) in enumerate(zip(means, variances)):

    x_regular = np.random.normal(mean, var, 100)
    y_regular = a * x_regular + b + np.random.normal(mean, np.sqrt(var), 100)

    x_highx = np.random.normal(mean, var * 3, 100)
    y_highx = a * x_highx + b + np.random.normal(mean, np.sqrt(var), 100)

    x_highy = np.random.normal(mean, var, 100)
    y_highy = a * x_highy + b + np.random.normal(mean, np.sqrt(var * 5), 100)

    x_both = np.random.normal(mean, var * 3, 100)
    y_both = a * x_both + b + np.random.normal(mean, np.sqrt(var * 5), 100)

    x = np.concatenate([x_regular, x_highx, x_highy, x_both])
    y = np.concatenate([y_regular, y_highx, y_highy, y_both])

    X = np.column_stack((np.ones_like(x), x))
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    H = U @ U.T
    leverage = np.diag(H)

    top_points = np.argsort(leverage)[-15:]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='steelblue', alpha=0.5, label='Normal points')
    plt.scatter(x[top_points], y[top_points], color='red', label='High leverage')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'1D Model: μ={mean}, σ²={var}')
    plt.legend()
    plt.tight_layout()

    plt.savefig("ex1_1.svg")

