import numpy as np #importing numpy for numerical operations
import matplotlib.pyplot as plt
#this is a simple implementation of a neural network from scratch using numpy and its sequential model. The neural network consists of an input layer, two hidden layers, and an output layer. The activation function used is the relu function.

#function to activate relu
def relu(x):
    return np.maximum(0, x)
#function to define the derivative of relu
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Dataset
X = np.array([[5], [10], [15], [20], [25], [30], [35], [40]])
y = np.array([[35], [42], [48], [58], [63], [72], [78], [88]])

# Z-score scaling for X
mu = np.mean(X)
sigma = np.std(X)
X_scaled = (X - mu) / sigma

print(X_scaled.shape)
print(y.shape)

# Initial weights
w1 = np.array([[0.5]])
w2 = np.array([[0.8]])
w3 = np.array([[1.2]])

# Initial biases
b1 = np.array([[1.0]])
b2 = np.array([[0.5]])
b3 = np.array([[1.0]])

# Learning rate (tuned for scaled X and raw y)
alpha = 0.002
#List to keep track of loss to plot it
loss_history = []

for epoch in range(2000):
    #FORWARD PROPAGATION
    #neurons will be defined by z = w*x + b
    z1 = np.dot(X_scaled, w1) + b1 #first neuron 
    z2 = np.dot(z1, w2) + b2 #second neuron using the output of the first neuron
    a2 = relu(z2) #activation function 
    z3 = np.dot(a2, w3) + b3 #third neuron using the output of the second neuron activation function
    y_hat = z3 #output of the neuron
    
    loss = np.mean((y_hat - y)**2) #loss function
    loss_history.append(loss)

    # BACKPROPAGATION
    error = (2 / len(y)) * (y_hat - y) #averaged error in here we divide by 8 to prvent the error from being to large   

    #calculating the gradient of the loss function with respect to the weights and biases
    #y_hat = w3 * a2 + b3 we are passing down w3 using the output of the second neuron activation function
    g_w3 = np.dot(a2.T, error) #gradient of the loss function with respect to the weights of the third neuron
    g_b3 = np.sum(error, axis=0, keepdims=True) #gradient of the loss function with respect to the biases of the third neuron gradient of b = 1 beacause gradeint of the loss function with respect to the biases is the same as the gradient of the loss function with respect to the output of the neuron
    #z2 = w2 * z1 + b2 
    #relu gradient
    relu_g = np.dot(error, w3.T) * relu_derivative(z2) #layer_error = (error * previous_weight) * relu
    g_w2 = np.dot(z1.T, relu_g) #gradient of the loss function with respect to the weights of the second neuron
    g_b2 = np.sum(relu_g, axis=0, keepdims=True) #gradient of the loss function with respect to bias of the second neuron

    g_w1 = X_scaled.T @ (relu_g @ w2.T)  #gradient of the loss function with respect to the weights of the first neuron
    g_b1 = np.sum(relu_g @ w2.T, axis=0, keepdims=True) #gradient of the loss function with respect to the biases of the first

    #applying the gradient decent to the weights and biases
    w1 = w1 - alpha * g_w1
    b1 = b1 - alpha * g_b1

    w2 = w2 - alpha * g_w2
    b2 = b2 - alpha * g_b2

    w3 = w3 - alpha * g_w3
    b3 = b3 - alpha * g_b3

    if epoch % 400 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print("\nFinal predictions:\n", np.round(y_hat.flatten(), 2))
print("Actual targets:\n", y.flatten())

#plotting the loss over epochs
plt.plot(loss_history)
plt.title("Training Loss Over Epochs (Z-Score Scaled X)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()

# 1. Define the new unseen input as a 2D array (1 sample, 1 feature)
X_new = np.array([[45]])

# 2. Scale it using the exact same mean and std from your training data
X_new_scaled = (X_new - mu) / sigma

# 3. Run the forward pass step-by-step through each layer
z1_new = np.dot(X_new_scaled, w1) + b1
z2_new = np.dot(z1_new, w2) + b2
a2_new = relu(z2_new)
z3_new = np.dot(a2_new, w3) + b3

# 4. Extract the final scalar prediction
pred_new = z3_new[0][0]
print(f"Prediction for X=45: {pred_new:.2f}")