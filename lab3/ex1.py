import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500,
                  n_features=2,
                  centers=None,
                  cluster_std=1.0,
                  random_state=42)

num_proj = 5
projections = []

for _ in range(num_proj):
    v = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2))
    v = v / np.linalg.norm(v)   
    projections.append(v)

num_bins = 20

histograms = []
bin_edges_list = []

for v in projections:
    proj_vals = X.dot(v)  
    
    data_min, data_max = proj_vals.min(), proj_vals.max()
    hist_range = (data_min - 1, data_max + 1)
    
    counts, bin_edges = np.histogram(proj_vals, bins=num_bins, range=hist_range)
    
    probs = counts / counts.sum()
    
    histograms.append(probs)
    bin_edges_list.append(bin_edges)

X_test = np.random.uniform(low=-3, high=3, size=(500, 2))

scores = []

for x in X_test:
    p_vals = []
    for v, probs, edges in zip(projections, histograms, bin_edges_list):
        proj = x.dot(v)
        
        bin_idx = np.searchsorted(edges, proj) - 1
        if bin_idx < 0 or bin_idx >= len(probs):
            p_vals.append(0)
        else:
            p_vals.append(probs[bin_idx])
    
    scores.append(np.mean(p_vals))

scores = np.array(scores)

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")
plt.colorbar(label="anomaly score")
plt.title(f"LODA slike score map ({num_bins} bins)")
plt.show()
