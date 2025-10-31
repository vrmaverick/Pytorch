import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X,y = datasets.make_regression(n_samples = 100,n_features = 1,noise = 20,random_state = 69)
X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))
y_tensor = y_tensor.view(-1,1) #it is in a list of 100 => list 100 vecors/sublists

n_samples, n_features = X_tensor.shape
print(n_samples,n_features)
# print(y_tensor)
# Model
model = nn.Linear(n_features,1,bias = True)

# Loss and Optimizer
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
num_epoches =100
for epoch in range(num_epoches):
  y_pred = model(X_tensor)
  l = loss(y_tensor,y_pred)

  l.backward()

  optimizer.step() # for updation

  optimizer.zero_grad() #Empty our gradients

  if epoch % 10 == 0:
    # [w,b] = model.parameters()
   print(f'epoch {epoch+1}: loss = {l:.8f}')


# plotting
predicted = model(X_tensor).detach().numpy() # detach as we need to prevent this operation in computational graph
#  tensor from  requires_grad = false
plt.plot(X_tensor.numpy(),y_tensor.numpy(),'ro')
plt.plot(X_tensor.numpy(),predicted, 'b')
plt.show()

