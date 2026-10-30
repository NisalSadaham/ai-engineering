from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_actual = [70, 80, 90, 60]
y_predicted = [72, 77, 89, 62]

#calculate mean absolute error

calculate_mae = mean_absolute_error(y_pred=y_predicted,y_true=y_actual)
print("MAE:",calculate_mae)


#calcualte mean squared error

calculate_mse = mean_squared_error(y_pred=y_predicted,y_true=y_actual)
print("MSE:",calculate_mse)

#calculate root mean squared error4

rmse = np.sqrt(calculate_mae)
print("RMSE:",rmse)


