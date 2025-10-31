# concept of local gradient stored during modelling as
f(x,y) = x*y
local gradient is d(x*y)/d(x) helps calculateing d(loss)/d(x) -- >we need to minimize it

# we will do it manually first then we will use functions
import numpy as np
import torch
X = torch.tensor([1,2,3,4,5,6,7,8,9])
# X = torch.tensor([1,2,3,4,5,6,7,8,9],dtype = torch.float32,requires_grad = True)
Y = torch.tensor([3,5,7,9,11,13,15,17,19])
# Y = torch.tensor([3,5,7,9,11,13,15,17,19],dtype = torch.float32,requires_grad = True)

w = 0.0
b = 0

print(X)
print(Y)

# Forward pass model prediction
def forward(x) :
  return w*x + b
# Loss
def Loss (y,y_pred):
  return ((y_pred-y)**2).mean()

# Gradient
# dJ/dw = 1/N 2x(w*x+b-y)
# dJ/db = 1/N 2(w*x+b-y)
def gradient(x,y,y_pred):
  dw = np.dot(2*x , y_pred-y).mean()
  db = np.dot(2 , y_pred-y).mean()
  return dw,db

print(f'Prediction before training : f(5) = {forward(5):.3f}')
learning_rate = 0.001
n_iters = 20

for epoch in range(n_iters):
  y_pred = forward(X)
  l = Loss(Y,y_pred)
  dw,db = gradient(X,Y,y_pred)

  # update the weights in GD algorithm is going to negative direction
  w -= learning_rate * dw
  b -= learning_rate * db

  if epoch % 2 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

X = torch.tensor([1,2,3,4,5,6,7,8,9],dtype = torch.float32)
Y = torch.tensor([3,5,7,9,11,13,15,17,19],dtype = torch.float32)
w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)
b = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# Forward pass model prediction
def forward(x) :
  return w*x + b
# Loss
def Loss (y,y_pred):
  return ((y_pred-y)**2).mean()

# Gradient
# dJ/dw = 1/N 2x(w*x+b-y)
# dJ/db = 1/N 2(w*x+b-y)
def gradient(x,y,y_pred):
  dw = np.dot(2*x , y_pred-y).mean()
  db = np.dot(2 , y_pred-y).mean()
  return dw,db

print(f'Prediction before training : f(5) = {forward(5):.3f}')
learning_rate = 0.01
n_iters = 35

for epoch in range(n_iters):
  y_pred = forward(X)
  l = Loss(Y,y_pred)

  # dw,db = gradient(X,Y,y_pred)
  l.backward()

  # update the weights in GD algorithm is going to negative direction
  # we need to make this part of computational function in gradient
  with torch.no_grad() :
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

  # clear old gradients as they will else be summed up
  w.grad.zero_()
  b.grad.zero_()

  if epoch % 2 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, b = {b:.3f} loss = {l:.8f}')

    # Not good performance as backpropogation is not that numerically correct as we did in mannual method


