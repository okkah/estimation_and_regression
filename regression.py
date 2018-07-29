from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import linear_model, grid_search
import matplotlib.pyplot as plt

X = np.linspace(-10, 10, 20)
Y = 0.001 * (X*X*X + X*X + X) + np.random.normal(0, 0.1, len(X))
poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X[:, np.newaxis])

parameters1 = {'alpha' : np.logspace(-3, 1, 100)}
parameters2 = {'alpha' : np.logspace(1, 4, 100)}
parameters3 = {'alpha' : np.logspace(-3, 1, 100), 'l1_ratio' : np.logspace(-1, 0, 10)}

model_l1 = grid_search.GridSearchCV(linear_model.LassoLars(), parameters1, cv=10)
model_l1.fit(X_poly, Y)

model_l2 = grid_search.GridSearchCV(linear_model.Ridge(), parameters2, cv=10)
model_l2.fit(X_poly, Y)

xs = np.linspace(-10, 10, 200)
Y_predict_1 = model_l1.predict(poly.fit_transform(xs[:, np.newaxis]))
Y_predict_2 = model_l2.predict(poly.fit_transform(xs[:, np.newaxis]))

print("Lasso Regresion : ")
print(model_l1.best_params_)
print(model_l1.best_estimator_.coef_)

print("Ridge Regresion : ")
print(model_l2.best_params_)
print(model_l2.best_estimator_.coef_)

plt.plot(X, Y, ".", color="k")
plt.plot(xs, Y_predict_1, "-", color="r", label="Lasso")
plt.plot(xs, Y_predict_2, "-", color="b", label="Ridge")
plt.legend(loc="lower left")
plt.show()
