#Implementing linear regression using scikit learn
import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression #imported linear regression from scikit-learn
from sklearn.model_selection import train_test_split #importing function to split the data and train them
from sklearn.metrics import mean_absolute_error #importing mean error
#It measures the average absolute difference between actual values and predictions.

x = np.array([[1],[2],[3],[4],[5]]) #notice that the feature has [1] unlike [1,2,3,4] on which i made from scratch:because sk-learn expects feature to have a shape(5 samples 1 feature)
print(x.shape)
y = np.array([3,5,9,15,23]) #actual results

model = LinearRegression() #linear regression instance

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
) #saying to use 20% of the data for testing ,random_state so that it reproduce the same split

model.fit(X=x_train,y=y_train) #fitting the features and the actual result into the model 
prediction = model.predict(X=x_test)
mean_error = mean_absolute_error(y_test, prediction)
print("Mean absolute error:", mean_error)
print(prediction)
print("Coefficient:", model.coef_) #gives the value of the parameter w(weight)
print("Intercept:", model.intercept_)#gives the value of the parameter b(bias)

mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

print("Test MSE:", mse)
print("Test RMSE:", rmse)