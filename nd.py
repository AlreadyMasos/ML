from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

knn = KNeighborsClassifier(n_neighbors=1)
iris_dataset = load_iris()


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
knn.fit(X_train, y_train)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

"""new = np.array([[4, 2.5, 0.9, 0.1]])
predict = knn.predict(new)
print(f'Прогноз {predict}')
print('Метка {}'.format(iris_dataset['target_names'][predict]))"""

y_pred = knn.predict(X_test)
print(np.mean(y_pred == y_test))