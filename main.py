import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.keys())
# print(len(diabetes.data))
diabetes_x = diabetes.data[:, np.newaxis, 2]
# print(diabetes_x)
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predict = model.predict(diabetes_x_test)

print("Mean Squad Error: ", mean_squared_error(diabetes_y_test, diabetes_y_predict))
print("Weights: ", model.coef_)
print('Intercept: ', model.intercept_)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predict)
plt.show()

# Mean Squad Error:  3035.0601152912686
# Weights:  [941.43097333]
# Intercept:  153.39713623331698