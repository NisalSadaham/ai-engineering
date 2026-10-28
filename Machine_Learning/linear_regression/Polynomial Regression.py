#polynomial regression from scratch
#will be using the same formulas as the multi_linear_regression
import numpy as np #using numpy for calculations
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5]) #feature
y = np.array([3, 5, 9, 15, 23]) #acutal outputs
x2 = x ** 2  # engineered feature: x squared
m = x.size
x_both = np.column_stack([x,x2]) #to combine both the features and features sqred
print(x_both.shape)  

w = np.zeros(2) #starting w = [0,0]
b = 0 #staring bias
alpha = 0.7#learning rate

iterations = 100

#function to scale down the features for polynomial regression
def feature_scaling(x_both):
    mu = np.mean(x_both,axis=0) # mean of columns
    sigma = np.std(x_both,axis=0) #std of columns
    x_scaled = (x_both - mu) / sigma
    return x_scaled,mu,sigma



def predictions(x_scaled,b,w):
    y_hat = np.dot(x_scaled,w) + b #predicition equation
    return y_hat

def error_calculation(y,y_hat):
    error = y_hat - y
    return error

def cost_function(error):
    error_sqr = error ** 2
    j = 1/(2*m) * sum(error_sqr)
    return j

def calculate_gradient(x_scaled,error):
    dj_dw =  1/m * np.dot(x_scaled.T,error) 
    dj_db = 1/m * sum(error)

    return dj_dw,dj_db

def gradient_decent(dj_dw,dj_db,b,w):
    w = w -alpha * dj_dw
    b = b -alpha * dj_db

    return w,b

def find_best_parameter(w,b,iteration,x_scaled):
    cost_history = []
    for i in range(iteration):
        y_hat = predictions(x_scaled,b,w)
        error = error_calculation(y,y_hat)
        cost = cost_function(error)
        print("Cost:",cost)
        cost_history.append(cost)
        dj_dw,dj_db = calculate_gradient(error=error,x_scaled=x_scaled)
        w,b = gradient_decent(w=w,b=b,dj_db=dj_db,dj_dw=dj_dw)
    return w,b,cost_history

def plot_cost_overtime(iterations,cost_history):
    plt.plot(range(iterations),cost_history,marker = ".",linestyle = '-')
    plt.title("Cost Reduction over Gradient Descent Iterations")
    plt.xlabel("iterations")
    plt.ylabel("Cost J(w,b)")
    plt.grid(True)
    plt.show()

def predictions_using_trained_model(trained_w,trained_b,mu,sigma):
    x_p = []
    x1 = int(input("Enter the value for feature(x1):"))
    x2 = x1 ** 2
    x_p.append(x1)
    x_p.append(x2)
    x_p_array = np.array(x_p)
    scal_x_p_array = (x_p_array - mu) / sigma
    prediction = np.dot(scal_x_p_array,trained_w) + trained_b
    return prediction

def main():
    x_scaled,mu,sigma = feature_scaling(x_both=x_both)
    trained_w,trained_b,cost_history = find_best_parameter(w=w,b=b,x_scaled=x_scaled,iteration=iterations)
    print(trained_w)
    print(trained_b)
    plot_cost_overtime(iterations=iterations,cost_history=cost_history)
    prediction = predictions_using_trained_model(trained_w=trained_w,trained_b=trained_b,mu=mu,sigma=sigma)
    print("Prediction:",prediction)

if __name__ == "__main__":
    main()
