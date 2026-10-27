#Linear regression with more than one feature
#Two features will be used ,x1 and x2
#Same as before w = weight y = actual result x = features/inputs
import numpy as np #will use numpy for array calculations
import matplotlib.pyplot as plt #to draw graphs

x1 = np.array([2,4,6,8]) #feature 1 study hours
x2 = np.array([10,20,30,40])# feature 2 Practice hours
y = np.array([45,55,70,80])#actual results
m = y.size
iterations = 1000
alpha = 0.1

x = np.column_stack([x1,x2])#combining both the x1 and x2 features to create a 2d matrix 
#print(x)
b = 50 #starting bias
w = np.zeros(x.shape[1])#starting weights for both features
#print(w.shape)
#print(x.shape)
#print(y.size)
#function to scale the data set for linear regression
def feature_scaling(x):
    mu = np.mean(x,axis=0) #taking the mean of the columns
    sigma = np.std(x,axis=0) #taking the standard deviation of the columns
    x_scaled = (x - mu) / sigma #using z score formula to scale the dataset
    return x_scaled,mu,sigma #scaled values of x
#print(feature_scaling(x=x).shape)
#print(w)
#function to make predictions using the scaled features
def predictions(x_scaled,w,b):
    #formula changes a bit because of the n of features
    y_hat = np.dot(x_scaled,w) + b #here using a dot multiplication because now there are two features and w weights corresponding to them.(4x2)(2x1) = 4x1
    return y_hat #returns the result

#function to calculate the error
def calculate_error(y_hat,y):
    error = y_hat - y
    return error

#calculating the cost of the model
def cost_function(error):
    er_sqr = error ** 2 #square the error
    j = 1/(m * 2) * sum(er_sqr) #cost function 
    return j
#function to find new gradients for w,b
def calculate_gradient(x_scaled,error):
    dj_dw = 1/m  * np.dot(x_scaled.T,error)#gradient of w got by multipltying transpose of x by error(feature x error)
    dj_db = 1/m  * sum(error)#gradient of bias
    #print("check",dj_db)
    #print("sum",sum(x_scaled))
    return dj_dw,dj_db
#function to update the parameters using gradient decent algo
def gradient_decent(dj_dw,dj_db,w,b):
    w = w - alpha * (dj_dw)#update parameters for w
    #print("w",w)
    b = b - alpha * (dj_db)#update new paramater for b
    #print("b",b)
    return w,b
def find_best_parameters(w,b,iterations,x_scaled):
    cost_history = []
    for i in range(iterations):
        y_hat = predictions(x_scaled=x_scaled,w=w,b=b)
        #print(y_hat)
        error = calculate_error(y_hat=y_hat,y=y)
        #print(error)
        cost = cost_function(error=error)
        #print("Cost",cost)
        cost_history.append(cost)#to store the cost of the model
        #print(cost_history)
        dj_dw,dj_db = calculate_gradient(x_scaled=x_scaled,error=error)
        #print(dj_dw)
        #print(dj_db)
        w,b = gradient_decent(dj_db=dj_db,dj_dw=dj_dw,w=w,b=b)     
    return w,b
#function to plot a graph for the costs calculated
def plot_graph_cost(cost_history,iterations):
    plt.plot(range(iterations),cost_history,marker = ".",linestyle = '-')
    plt.title("Cost Reduction over Gradient Descent Iterations")
    plt.xlabel("iterations")
    plt.ylabel("Cost J(w,b)")
    plt.grid(True)
    plt.show()
def predictions_using_trained_model(trained_w,trained_b,mu,sigma):
    x_p = []
    x1 = int(input("Enter the value for feature 1(x1):"))
    x2 = int(input("Enter the value for feature 2(x2):"))
    x_p.append(x1)
    x_p.append(x2)
    x_p_array = np.array(x_p)
    scal_x_p_array = (x_p_array - mu) / sigma
    prediction = np.dot(scal_x_p_array,trained_w) + trained_b

    return prediction
def main():
    x_scaled,mu,sigma = feature_scaling(x=x)
    trained_w,trained_b = find_best_parameters(w=w,b=b,iterations=iterations,x_scaled=x_scaled)
    #plot_graph_cost(cost_history=cost_history,iterations=iterations)
    #print(trained_w)
    #print(trained_b)
    prediction = predictions_using_trained_model(trained_b=trained_b,trained_w=trained_w,mu=mu,sigma=sigma)
    print("Prediction:",prediction)
main()
