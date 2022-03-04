
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#lr = LinearRegression().fit(X_train, y_train)
#print(lr.score(X_train, y_train))
#print(lr.score(X_test, y_test))
#чем меньше альфа - тем сложнее модель, тем больше она похожа на линейную регрессию
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print(round(ridge.score(X_train, y_train),2))
print(round(ridge.score(X_test, y_test),2))
#обратный случай
ridge = Ridge(alpha=10).fit(X_train, y_train)
print(round(ridge.score(X_train, y_train),2))
print(round(ridge.score(X_test, y_test),2))
#чем меньше alpha в lasso, ием меньше признаков обращается в ноль
lasso = Lasso(alpha=0.01,max_iter=10000).fit(X_train, y_train)
print(round(lasso.score(X_train, y_train),2))
print(round(lasso.score(X_test, y_test),2))
print(np.sum(lasso.coef_ != 0))