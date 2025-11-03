import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

data = scipy.io.loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_models = 10
neighbors_values = np.linspace(30, 120, n_models, dtype=int)

train_scores_list = []
test_scores_list = []

for n_neighbors in neighbors_values:
    model = KNN(n_neighbors=n_neighbors, contamination=0.1, n_jobs=1)
    model.fit(X_train_scaled)

    train_scores = model.decision_scores_
    test_scores = model.decision_function(X_test_scaled)

    train_scores_list.append(train_scores)
    test_scores_list.append(test_scores)

    threshold = np.quantile(train_scores, 0.9)
    y_train_pred = (train_scores > threshold).astype(int)
    y_test_pred = (test_scores > threshold).astype(int)

    print(f"KNN n_neighbors={n_neighbors}")
    print("Train BA:", balanced_accuracy_score(y_train, y_train_pred))
    print("Test BA :", balanced_accuracy_score(y_test, y_test_pred))
    print("-" * 35)

train_scores_arr = np.array(train_scores_list).T
test_scores_arr = np.array(test_scores_list).T

train_scores_norm, test_scores_norm = standardizer(train_scores_arr, X_t=test_scores_arr)

combined_avg = average(test_scores_norm)
combined_max = maximization(test_scores_norm)

contamination = 0.1

threshold_avg = np.quantile(combined_avg, 1 - contamination)
y_pred_avg = (combined_avg > threshold_avg).astype(int)
print("Ensemble Average BA:", balanced_accuracy_score(y_test, y_pred_avg))

threshold_max = np.quantile(combined_max, 1 - contamination)
y_pred_max = (combined_max > threshold_max).astype(int)
print("Ensemble Maximization BA:", balanced_accuracy_score(y_test, y_pred_max))
