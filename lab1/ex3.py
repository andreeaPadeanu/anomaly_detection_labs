import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import  balanced_accuracy_score

np.random.seed(42) 
#i used a random seed here just to keep the results stable every time i run the code

X_train, _, y_train, _ = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=0.1
)

z = np.abs((X_train - X_train.mean()) / X_train.std())
th = np.quantile(z, 1 - 0.1)
y_pred = (z > th).astype(int)
bal_acc = balanced_accuracy_score(y_train, y_pred)
print(th, bal_acc)
