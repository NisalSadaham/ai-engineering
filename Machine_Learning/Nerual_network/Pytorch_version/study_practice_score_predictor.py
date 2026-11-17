#Same data set used in the neural network from scratch will be used
import torch

X = torch.tensor([[5,10], [10,15], [1,10], [10,13], [15,3], [20,5]],dtype=torch.float32)
X2 = torch.tensor([[5,10], [10,15], [1,10], [10,13], [15,3], [20,5]],dtype=torch.float64)

print(X.shape)
print(X.dtype)
print(X.element_size())
print(X2.element_size())