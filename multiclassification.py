import mglearn
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
X, y = make_blobs(random_state=42)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

linear_svm = LinearSVC().fit(X,y)
#coef_ имеет форму (3,2), это означает, что каждая строка coef_ содержит ветктор
#коэффицентов для каждого из трех классов, а каждый столбец содержит значнеие
#коэффицента для каждого признака(в этом наборе их два)
print(linear_svm.coef_.shape)
