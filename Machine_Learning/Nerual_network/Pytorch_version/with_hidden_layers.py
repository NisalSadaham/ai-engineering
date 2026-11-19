import torch
import torch.nn as nn
import torch.optim

layer = nn.Linear(4,8) #created a hidden layer of 4 features and 8 neurons
output_layer = nn.Linear(8,1) #created an output layer

#print(layer.weight)
#print(layer.bias)
#print(layer.weight.shape)
#print(layer.bias.shape)


x = torch.tensor([
    [2.0, 5.0, 3.0, 7.0],
    [1.0, 4.0, 6.0, 8.0],
    [3.0, 6.0, 2.0, 5.0],
    [4.0, 3.0, 7.0, 6.0],
    [5.0, 2.0, 4.0, 9.0]
])

y = torch.tensor([
    [45.0],
    [55.0],
    [65.0],
    [75.0],
    [85.0]
])
relu = nn.ReLU()#have to initialze first
calculate_loss = nn.MSELoss()
optimizer = torch.optim.SGD(
    list(layer.parameters()) + list(output_layer.parameters()),
    lr=0.001
)
for epochs in range(1000):
    z = layer(x)                  # 1. Linear transformation
    a = relu(z)                   # 2. Activation
    y_hat = output_layer(a)       # 3. Predictions
    loss = calculate_loss(y_hat, y)  # 4. Calculate error
    optimizer.zero_grad()         # 5. Clear old gradients
    loss.backward()               # 6. Calculate gradients
    optimizer.step()              # 7. Update weights and biases

print("Final predictions:", y_hat)
print("Actual values:", y)
print("Final loss:", loss.item())



print("Hidden layer weights:")
print(layer.weight)

print("Hidden layer biases:")
print(layer.bias)

print("Output layer weights:")
print(output_layer.weight)

print("Output layer biases:")
print(output_layer.bias)

print("Final hidden activations:")
print(a)