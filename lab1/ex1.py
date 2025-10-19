import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1
)

plt.figure(figsize=(6, 5))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], s=20, c="green", label="normal")
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], s=20, c="orange", label="anomaly")
plt.title("detected anomalies")
plt.legend()
plt.tight_layout()
plt.savefig("fig1.svg")
plt.show()

