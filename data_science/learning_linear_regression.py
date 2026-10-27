import numpy as np 
#asumming weight,w = 3,bias,b = 0 assuming total perdictions = 3
#i is used as x in the loop 
w = 3
b = 0
#prediction = 3 * x + b
total_prediction = 3
for_loops_prediction = total_prediction+1

y_real = [2,4,6]
y_prediction_list = []

# calculate error
for i in range(1,for_loops_prediction):
    y_prediction = w * i + b
    y_prediction_list.append(y_prediction)


y_prediction_list = np.array(y_prediction_list)
y_real  = np.array(y_real)

error = y_prediction_list - y_real
error_squared = error ** 2
cost = (1/(y_prediction_list.size*2)) * sum(error_squared)
print(f"Cost:{cost}")




    

