import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('traffic_index.csv')

df.dropna(inplace=True)

y = df['average_traffic_index']
X = df.drop('average_traffic_index',axis=1)


X_train,X_test,y_train,y_test =  train_test_split(random_state=30,test_size=0.2)

#Linear Regression

lr = LinearRegression()
lr.fit(X_train,y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred  = lr.predict(X_test)




#testing the models
lr_mse_train_pred = mean_squared_error(y_train,y_lr_train_pred)
lr_r2_train_pred = r2_score(y_train,y_lr_train_pred)

lr_mse_test_pred = mean_squared_error(y_test,y_lr_test_pred)
lr_r2_test_pred = r2_score(y_test,y_lr_test_pred)


#Random Forest

rf = RandomForestRegressor(random_state=30)
rf.fit(X_train,y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)



#testing the models

rf_mse_train_pred = mean_squared_error(y_train,y_rf_train_pred)
rf_r2_train_pred = r2_score(y_train,y_rf_train_pred)

rf_mse_test_pred = mean_squared_error(y_test,y_rf_test_pred)
rf_r2_test_pred = r2_score(y_test,y_rf_test_pred)


#printing out the results


#Linear Regression Results
print('Lineear regression MSE:', lr_mse_test_pred, lr_mse_train_pred)
print('Linear Regression R2:', lr_r2_train_pred,lr_r2_test_pred)

#Random Forest Results
print('Random Forest MSE:', rf_mse_train_pred,rf_mse_test_pred)
print('Random Forest R2:', rf_r2_train_pred,rf_r2_test_pred)

#visualising the results gathered
plt.title('Linear regression')
plt.xlabel('testing models')
plt.ylabel('Test scores')
plt.scatter(lr_mse_test_pred,lr_mse_train_pred)
plt.show()

plt.title('Random Forest')
plt.xlabel('testing models evaluated')
plt.ylabel('testing model scores')
plt.scatter(rf_mse_test_pred,rf_mse_train_pred)
plt.show()


plt.title('Random Forest VS Linear Regression')
plt.xlabel('testing models evaluated')
plt.ylabel('testing model scores')
plt.scatter(rf_r2_test_pred,lr_r2_test_pred)
plt.show()


