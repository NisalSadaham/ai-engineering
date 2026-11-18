import torch
import torch.nn as nn
import torch.optim

layer = nn.Linear(4,8) #created a hidden layer of 4 features and 8 neurons

#print(layer.weight)
#print(layer.bias)
#print(layer.weight.shape)
#print(layer.bias.shape)


x = torch.tensor([
    [2.0, 5.0, 3.0, 7.0],
    [1.0, 4.0, 6.0, 8.0]
])

output = layer(x)

print(output)
print(output.shape)