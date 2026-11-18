import torch
import torch.optim



Epochs = 1000
x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]]) #requires_grad=True is used to enable gradient computation and to track it
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y = torch.tensor([[5.0],[8.0],[11.0],[14.0],[17.0]])

#torch.optim is a module that provides various optimization 
optimizer = torch.optim.SGD([w,b],lr=0.01)
for i in range (Epochs):
    optimizer.zero_grad() #zero_grad() is used to clear the gradients of the previous gradeint calculation
    y_hat = w * x + b #y is a tensor which is a function of x sample linear equation
    loss = ((y_hat - y)**2).mean() #loss

    loss.backward()
    optimizer.step()

    if i % 10 == 0: 
        print("Updated weight:", w.item())
        print("Updated bias:", b.item())
        print("Loss:", loss.item())


#print(y_hat.item()) #y.item() is the value of y
#print(x.item()) #x.item() is the value of x

#print(x.grad.item()) #x.grad is gradient of y with respect to x
#print(w.grad.item())
#print(b.grad.item()) 

test_x = torch.tensor(4.0)
new_y_hat = w * test_x + b 
print("New y_hat:", new_y_hat.item())






