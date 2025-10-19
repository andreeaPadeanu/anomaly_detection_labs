from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, roc_auc_score
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1
)

clf = KNN(contamination=0.1).fit(X_train)
y_pred_train, y_pred_test = clf.predict(X_train), clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
bal_acc = balanced_accuracy_score(y_test, y_pred_test)

fpr, tpr, _ = roc_curve(y_test, clf.decision_function(X_test))
print("AUC:", roc_auc_score(y_test, clf.decision_function(X_test)))
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, c="blue")
plt.plot([0, 1], [0, 1], "--", c="pink")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig("fig2.svg")
plt.show()

print(tn, fp, fn, tp, bal_acc)
