import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from pyod.utils.utility import standardizer

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

n_runs = 10
ba_if, auc_if = [], []
ba_dif, auc_dif = [], []
ba_loda, auc_loda = [], []

for _ in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, shuffle=True, stratify=y
    )

    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    iforest = IForest(contamination=0.02, random_state=42)
    iforest.fit(X_train_norm)
    scores_if = iforest.decision_function(X_test_norm)

    dif = DIF(contamination=0.02,
              hidden_neurons=[16, 8],
              device="cpu",
              random_state=42)
    dif.fit(X_train_norm)
    scores_dif_model = dif.decision_function(X_test_norm)

    loda = LODA(contamination=0.02, n_bins=20)
    loda.fit(X_train_norm)
    scores_loda_model = loda.decision_function(X_test_norm)

    ba_if.append(balanced_accuracy_score(y_test, iforest.predict(X_test_norm)))
    auc_if.append(roc_auc_score(y_test, scores_if))

    ba_dif.append(balanced_accuracy_score(y_test, dif.predict(X_test_norm)))
    auc_dif.append(roc_auc_score(y_test, scores_dif_model))

    ba_loda.append(balanced_accuracy_score(y_test, loda.predict(X_test_norm)))
    auc_loda.append(roc_auc_score(y_test, scores_loda_model))

print("Isolation Forest:       BA =", np.mean(ba_if),  "AUC =", np.mean(auc_if))
print("Deep Isolation Forest:  BA =", np.mean(ba_dif), "AUC =", np.mean(auc_dif))
print("LODA:                   BA =", np.mean(ba_loda), "AUC =", np.mean(auc_loda))
