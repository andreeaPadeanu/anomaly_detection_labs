import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

X, _ = make_blobs(n_samples=1000,
                  n_features=2,
                  centers=[[10, 0], [0, 10]],
                  cluster_std=1.0,
                  random_state=42)

test_data = np.random.uniform(low=-10, high=20, size=(1000, 2))

iforest = IForest(contamination=0.02,
                  n_estimators=100,
                  max_samples=256,
                  random_state=42)

iforest.fit(X)
scores_iforest = iforest.decision_function(test_data)


plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(test_data[:, 0], test_data[:, 1],
            c=scores_iforest, cmap="viridis")
plt.title("isolation forest")
plt.colorbar(label="anomaly score")

dif = DIF(contamination=0.02,
          hidden_neurons=[16, 8],   
          random_state=42)

dif.fit(X)
scores_dif = dif.decision_function(test_data)

plt.subplot(1, 3, 2)
plt.scatter(test_data[:, 0], test_data[:, 1],
            c=scores_dif, cmap="viridis")
plt.title("deep isolation forest")
plt.colorbar(label="anomaly score")

loda = LODA(contamination=0.02,
            n_bins=20)

loda.fit(X)
scores_loda = loda.decision_function(test_data)


plt.subplot(1, 3, 3)
plt.scatter(test_data[:, 0], test_data[:, 1],
            c=scores_loda, cmap="viridis")
plt.title("LODA")
plt.colorbar(label="anomaly score")

plt.tight_layout()
plt.show()

#everything in 3D
from mpl_toolkits.mplot3d import Axes3D  

X3, _ = make_blobs(n_samples=1000,
                   n_features=3,
                   centers=[[0, 10, 0], [10, 0, 10]],
                   cluster_std=1.0,
                   random_state=42)

test3 = np.random.uniform(low=-10, high=20, size=(1000, 3))

if3 = IForest(contamination=0.02, random_state=42)
if3.fit(X3)
scores3 = if3.decision_function(test3)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(test3[:, 0], test3[:, 1], test3[:, 2],
               c=scores3, cmap="viridis")
fig.colorbar(p, label="anomaly score")
ax.set_title("isolation forest in 3D")
plt.show()
