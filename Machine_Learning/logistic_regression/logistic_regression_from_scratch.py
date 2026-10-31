#Logisitic regression from scratch
import numpy as np #used for calculations
import matplotlib.pyplot as plt #to make graphs

#2 features are used here x1 = study hours and x2 = pervious score
X = np.array([
    [1, 35],
    [2, 50],
    [3, 45],
    [4, 60],
    [5, 55],
    [2, 65],
    [6, 70],
    [4, 45],
    [7, 80],
    [5, 75],
    [3, 60],
    [6, 50],
    [8, 85],
    [2, 40],
    [7, 65]
])

#actual values
y = np.array([
    0, 0, 0, 1, 0,
    1, 1, 0, 1, 1,
    1, 0, 1, 0, 1
])
#confirming the shapes to be used for calculations
#print(X.shape)#(15,2)
#print(y.shape)#(15,)

w = np.zeros(X.shape[1])#using the column size to declare weight paramater ([0,0])
b = 0 #bias paramater
print(w.shape)
alpha = 0.01 #The learning rate
iterations = 2000 #number of iterations the model will loop
m = X.shape[0]
#Function for feature scaling
def feature_scaling(X):
    mu = np.mean(X,axis=0) #mean of features
    sigma = np.std(X,axis=0) #standard deviation of features
    X_scaled = (X - mu) / sigma #scaled features
    return X_scaled,mu,sigma #tested and worked

#Function to get the linear predicitions
def linear_prediction(X_scaled,w,b):
    #print(X_scaled.shape)
    #print(w.shape)
    z = np.dot(X_scaled,w) + b #in here the x_scaled is multiplied by w because the inner dimentions much match
    return z

#Function to do the sigmoid calculation for the linear predicitions
def sigmoid_function(X_scaled,w,b):
    z = linear_prediction(X_scaled=X_scaled,b=b,w=w) #linear predicitions
    e = np.exp(-z) #to calculate the e constant to power (z)
    y_prob = 1/(1+e) #the sigmoid calculation
    return y_prob
#function to make predictions
def predicitions(y_prob):
    y_hat = np.where(y_prob >= 0.5,1,0) #set a boundary and get the results
    return y_hat
#function to calculate error
def calculate_error(y_prob,y):
    error = y_prob - y #predicitions - actual valuees
    return error 
#function to calculate the cost
def cost_function(y_prob,y,m):
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)#used to prevent log(0) errors
    loss = y * np.log(y_prob) + (1 - y) * np.log(1-y_prob) #calculating the loss
    j = -1/m * np.sum(loss) #calculating the cost
    return j
#function to calculate the gradients of paramaters w,b for gradient descent
def calculate_gradient(X_scaled,error,m):
    dj_dw = 1/m * np.dot(X_scaled.T,error)
    dj_db = 1/m * sum(error)
    return dj_dw,dj_db

#function to use gradient descent to update parameters w,b 
def calculate_gradient_descent(w,b,alpha,dj_dw,dj_db):
    w = w -(alpha) * dj_dw
    b = b -(alpha) * dj_db

    return w,b


#function to find best parameters for model
def find_best_para(w,b,alpha,X_scaled,iterations):
    cost_history = []
    for i in range(iterations):
        #print(X_scaled)
        y_prob = sigmoid_function(X_scaled=X_scaled,b=b,w=w)
        #print(y_prob)
        y_hat = predicitions(y_prob=y_prob)
        #print(y_hat)
        error = calculate_error(y_prob=y_prob,y=y) #have to use the y_prob here 
        cost = cost_function(y_prob=y_prob,m=m,y=y)
        cost_history.append(cost)
        dj_dw,dj_db = calculate_gradient(X_scaled=X_scaled,error=error,m=m)
        #print(dj_dw,dj_db)
        w,b = calculate_gradient_descent(w=w,b=b,dj_db=dj_db,dj_dw=dj_dw,alpha=alpha)
    return w,b,cost_history
#function to plot cost graph
def plot_cost_overtime(iterations,cost_history):
    plt.plot(range(iterations),cost_history,marker = ".",linestyle = '-')
    plt.title("Cost Reduction over Gradient Descent Iterations")
    plt.xlabel("iterations")
    plt.ylabel("Cost J(w,b)")
    plt.grid(True)
    plt.show()

def predictions_using_trained_model(trained_w,trained_b,mu,sigma):
    result = ""
    x_p = []
    x1 = int(input("Enter the value for feature(x1):"))
    x2 = int(input("Enter the value for feature(x2):"))
    x_p.append(x1)
    x_p.append(x2)
    x_p_array = np.array(x_p)
    scal_x_p_array = (x_p_array - mu) / sigma
    y_prob = sigmoid_function(X_scaled=scal_x_p_array,b=trained_b,w=trained_w)
    prediction = predicitions(y_prob=y_prob)
    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"
    return result

def main():
    X_scaled,mu,sigma = feature_scaling(X=X)
    trained_w,trained_b,cost_history = find_best_para(w=w,b=b,alpha=alpha,X_scaled=X_scaled,iterations=iterations)
    #print(cost_history)
    #plot_cost_overtime(iterations=iterations,cost_history=cost_history)
    y_hat = predictions_using_trained_model(trained_b=trained_b,trained_w=trained_w,mu=mu,sigma=sigma)
    print("Prediction:",y_hat)

if __name__ == "__main__":
    main()
    