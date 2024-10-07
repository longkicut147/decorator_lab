from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score


def accuracy(func):
    def wrapper(model, data):
        X_test, y_test = data
        y_predict = model.predict(X_test)
        accuracy = r2_score(y_test, y_predict)
        return accuracy
    return wrapper


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# model = LogisticRegression()
# model.fit(X_train, y_train)


model = LinearRegression()
model.fit(X_train, y_train)

@accuracy
def accuracy_decorator(model, data):
    pass

accuracy = accuracy_decorator(model, (X_test, y_test))
print(accuracy)