import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('traffic_index.csv')

# Drop missing values
df.dropna(inplace=True)

# Separate features and target
y = df['average_traffic_index']
X = df.drop(['average_traffic_index', 'trafficindexdate'], axis=1)  # Exclude the date column if not required

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size=0.2)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions for Linear Regression
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Metrics for Linear Regression
lr_mse_train_pred = mean_squared_error(y_train, y_lr_train_pred)
lr_r2_train_pred = r2_score(y_train, y_lr_train_pred)

lr_mse_test_pred = mean_squared_error(y_test, y_lr_test_pred)
lr_r2_test_pred = r2_score(y_test, y_lr_test_pred)

# Random Forest
rf = RandomForestRegressor(random_state=30)
rf.fit(X_train, y_train)

# Predictions for Random Forest
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Metrics for Random Forest
rf_mse_train_pred = mean_squared_error(y_train, y_rf_train_pred)
rf_r2_train_pred = r2_score(y_train, y_rf_train_pred)

rf_mse_test_pred = mean_squared_error(y_test, y_rf_test_pred)
rf_r2_test_pred = r2_score(y_test, y_rf_test_pred)

# Printing out the results
# Linear Regression Results
print('Linear Regression MSE (Train, Test):', lr_mse_train_pred, lr_mse_test_pred)
print('Linear Regression R2 (Train, Test):', lr_r2_train_pred, lr_r2_test_pred)

# Random Forest Results
print('Random Forest MSE (Train, Test):', rf_mse_train_pred, rf_mse_test_pred)
print('Random Forest R2 (Train, Test):', rf_r2_train_pred, rf_r2_test_pred)

# Visualizing the results
# Linear Regression and Random Forest MSE Comparison
plt.figure(figsize=(8, 6))
plt.title('Model MSE Comparison (Train vs Test)')
plt.bar(['LR Train', 'LR Test', 'RF Train', 'RF Test'], 
        [lr_mse_train_pred, lr_mse_test_pred, rf_mse_train_pred, rf_mse_test_pred], 
        color=['blue', 'blue', 'green', 'green'])
plt.ylabel('MSE')
plt.show()

# Linear Regression and Random Forest R2 Comparison
plt.figure(figsize=(8, 6))
plt.title('Model R2 Comparison (Train vs Test)')
plt.bar(['LR Train', 'LR Test', 'RF Train', 'RF Test'], 
        [lr_r2_train_pred, lr_r2_test_pred, rf_r2_train_pred, rf_r2_test_pred], 
        color=['blue', 'blue', 'green', 'green'])
plt.ylabel('R2 Score')
plt.show()

# Random Forest vs Linear Regression R2 Comparison on Test Set
plt.figure(figsize=(8, 6))
plt.title('Random Forest VS Linear Regression R2 Score (Test Set)')
plt.bar(['Random Forest Test', 'Linear Regression Test'], [rf_r2_test_pred, lr_r2_test_pred], color=['green', 'blue'])
plt.ylabel('R2 Score')
plt.show()
