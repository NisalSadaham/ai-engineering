#A program to find the best fitting paramater to minimize the cost function of linear regression
#we are assuming that b bais = 0 for this exercise
import numpy as np
#weight
w = 1
#bias
b = 0
#learning rate(alpha in formula)
a = 0.5

#input values
x = np.array([1,2,3])

#actual results
y = np.array([2,4,6])

#predictions formula
p = w * x
#error = (prediction - actual value)
error = p - y

#slope calculation
#s =  w *(-(a)) * 1/len(x)*(np.sum(error * x))
#print(sum(error*x))
for i in range (20):
    #recaculate the error with the new w
    p = w * x
    error = p - y
    slope = 1/len(x) * (sum(error * x))
    w = w - a *(1/len(x)*(sum(error * x)))
    print("Step no:",(i+1))
    print("Slope:",slope)
    print("W:",w)
    
