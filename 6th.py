import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
boston_df = pd.read_csv("HousingData.csv")
print("Linear Regression on Boston Housing Dataset")
X = boston_df[['RM']]
y = boston_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
plt.scatter(X_test, y_test, color='green', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('Linear Regression on Boston Housing Dataset')
plt.legend()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
auto_df = pd.read_csv("auto-mpg.csv")
print("Polynomial Regression on Auto MPG Dataset")
auto_df['horsepower'] = auto_df['horsepower'].replace('?', np.nan).astype(float)
auto_df.dropna(inplace=True) 
X = auto_df[['horsepower']] 
y = auto_df['mpg'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
PR_model = LinearRegression()
PR_model.fit(X_train_poly, y_train)
y_pred = PR_model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
plt.scatter(X_test, y_test, color='purple', label='Actual')
sorted_indices = X_test.squeeze().argsort()
plt.plot(
    X_test.iloc[sorted_indices],
    y_pred[sorted_indices],
    color='red',
    label='Predicted'
)
plt.xlabel('Horsepower')
plt.ylabel('MPG (Miles Per Gallon)')
plt.title('Polynomial Regression on Auto MPG Dataset')
plt.legend()
plt.show()