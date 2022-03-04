#файлик теста линейной регрессии

import mglearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression(C=100).fit(X_train, y_train)

print(logreg.score(X_test, y_test))
print(logreg.score(X_train, y_train))