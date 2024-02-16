import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset from a CSV file
df = pd.read_csv('FinlandHousePrice.csv')

# ==================== Linear Regression ====================
# Define the features (X) and the target (y) for Linear Regression
X_lr = df[['Upper quartile']]  # Features (independent variable)
y = df['Price per square meter(EUR/m2)']  # Target variable (dependent variable)

# Split the dataset for Linear Regression
X_train_lr, X_test_lr, y_train, y_test = train_test_split(X_lr, y, test_size=1/3, random_state=0)

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test_lr)

# Data visualization for Linear Regression
plt.scatter(X_test_lr, y_test, color='red')
plt.plot(X_test_lr, y_pred_lr, color='blue')
plt.title('Relationship between Upper Quartile and Price per Square Meter')
plt.xlabel('Upper Quartile Price (EUR/m2)')
plt.ylabel('Price per Square Meter (EUR/m2)')
plt.savefig('scatter_plot.png')
plt.show()

# Linear Regression - Performance metrics
mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)
print('Linear Regression - Performance metrics:')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# ==================== Random Forest & Decision Tree ====================
# Define the features for Random Forest and Decision Tree
X_rf_dt = df[['Upper quartile', 'Real price index', 'Number']]

# Split the dataset
X_train_rf_dt, X_test_rf_dt, y_train, y_test = train_test_split(X_rf_dt, y, test_size=0.25, random_state=0)

# Train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train_rf_dt, y_train)
y_pred_rf = rf_model.predict(X_test_rf_dt)

# Random Forest - Performance metrics
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rf)
print('Random Forest - Performance metrics:')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Train the Decision Tree Regression model
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X_train_rf_dt, y_train)
y_pred_dt = dt_model.predict(X_test_rf_dt)

# Decision Tree - Performance metrics
mae = mean_absolute_error(y_test, y_pred_dt)
mse = mean_squared_error(y_test, y_pred_dt)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_dt)
print('Decision Tree - Performance metrics:')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
