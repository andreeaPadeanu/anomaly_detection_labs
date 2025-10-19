import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score

X_train, _, y_train, _ = generate_data(
    n_train=1000,
    n_test=0,
    n_features=2,
    contamination=0.1
)

mu = np.array([2, -1])
Sigma = np.array([[1.0, 0.5], [0.5, 1.5]])
L = np.linalg.cholesky(Sigma) 
Y = X_train @ L.T + mu

z = np.linalg.norm((Y - Y.mean(axis=0)) / Y.std(axis=0), axis=1)
th = np.quantile(z, 1 - 0.1)
y_pred = (z > th).astype(int)
bal_acc = balanced_accuracy_score(y_train, y_pred)
print(th, bal_acc)

plt.scatter(Y[y_pred==0,0], Y[y_pred==0,1], s=15, c="green", label="normal")
plt.scatter(Y[y_pred==1,0], Y[y_pred==1,1], s=15, c="orange", label="anomaly")
plt.title(f"detected anomalies (balanced acc={bal_acc:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig("fig4.svg")
plt.show()
