from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

NB = MultinomialNB(alpha=0.1)
#наивные байесовские методы хорошо и быстро обучаются и применяются, в основном, для работы с текстом
#чем больше параметр альфа, тем проще модель и наоборот
NB.fit(X_train, y_train)

print(NB.score(X_test, y_test))
print(NB.score(X_train, y_train))