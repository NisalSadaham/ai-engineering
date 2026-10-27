#linear regression from scratch with one feature
#for reference w = weight b = bias y = actual value
#first start by importing numpy to do arrary calculations
#I will use mathplot show plot the convergence as well
import numpy as np
import matplotlib.pyplot as plt
#I will be using sample starting weight,actual result,and bias
w = 5
b = 20
x = np.array([1, 3, 5])#inputs
y = np.array([30, 50, 70])#actual results
m = len(x)
alpha = 0.01 #learning rate
iterations = 50
cost_history = []

#function to calculate error
def calculate_predictions(x,b,w):
#First predict using the linear regression prediction equation
    y_hat = w  * x + b
    return y_hat
#function to calculate the errors made by the model
def calculate_errors(y,y_hat):
    error = y_hat - y
    return error

#function to calculate the cost of the model
def cost_function(error):
    #using Andrew Ng's cost function
    error_sqr = error ** 2 #have to square the error values first
    j = 1/(2*m) * sum(error_sqr)
    return j
#function to calculate gradient for w,b
def calculate_gradient(error):
    g_w = 1/m * sum((error) * x)#gradient of w
    g_b = 1/m * sum(error)#gradient of b
    return g_w,g_b
#function to calculate the gradient descent and update the paramaters w,b
def calculate_gradient_descent(g_w,g_b,w,b):
    #update the values of parameters
    w = w -(alpha) * (g_w)#new value of w
    b = b -(alpha) * (g_b)# new value of w
    return w,b
#function to find the fitting pramaters to make the cost lower
def find_best_paras(w,b,iterations):
    for i in range(iterations):
        y_hat = calculate_predictions(x=x,b=b,w=w)
        error = calculate_errors(y=y,y_hat=y_hat)   
        cost = cost_function(error=error)
        cost_history.append(cost)#to store the cost values
        #print("COST:",cost)#check updated cost
        g_w,g_b = calculate_gradient(error=error)
        w,b = calculate_gradient_descent(g_b=g_b,g_w=g_w,w=w,b=b)#to store updated w,b
        #print("W:", w, "B:", b)
    return w,b
#function to plot a graph for the costs calculated
def plot_graph_cost(cost_history,iterations):
    plt.plot(range(iterations),cost_history,marker = ".",linestyle = '-')
    plt.title("Cost Reduction over Gradient Descent Iterations")
    plt.xlabel("iterations")
    plt.ylabel("Cost J(w,b)")
    plt.grid(True)
    plt.show()
#function to predict a value using the trained parameters note only works according to the training set
def predictions_using_trained_model(trained_w,trained_b,input_x):
    y_hat = trained_w * input_x + trained_b
    return y_hat
def main():
    trained_w,trained_b = find_best_paras(w=w,b=b,iterations=iterations)
    #plot_graph_cost(cost_history=cost_history,iterations=iterations)
    try:
        input_x = int(input("Enter a value for(x) to predict using the model:"))
    except ValueError:
        print("Please enter a number")
        return
    y_hat = predictions_using_trained_model(trained_b=trained_b,trained_w=trained_w,input_x=input_x)
    y_hat = round(y_hat)
    print("Prediction:",y_hat)

main()