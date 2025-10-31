import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target
n_samples, n_features = X.shape
print(n_samples,n_features)

X_train, y_train,X_test, y_test = train_test_split(X,y,test_size = 0.8, random_state = 69)
# X_test = X_test.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
ss = StandardScaler() # concerts the data into mean of 0 and variation is unit = length
X_train = ss.fit_transform(X_train)
y_train = ss.transform(y_train)

# Convert it into a tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- ASSUMPTION: X and y are properly defined NumPy arrays ---

# 1. Splitting the data:
# train_test_split returns: X_train, X_test, y_train, y_test
# Note: With test_size=0.8, X_train is 20% of the data (small), and X_test is 80% (large).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 69)


# 2. Scaling the features (X)
ss = StandardScaler()
X_train = ss.fit_transform(X_train) # Fit and transform on TRAINING features.

# ✅ CRITICAL FIX: Transform the test features (X_test) using the scaler.
X_test = ss.transform(X_test)


# 3. Converting to Tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


# 4. Correcting the shape of y_train and y_test for BCELoss output (from [N] to [N, 1])
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

# Assuming n_features is set using X_train.shape[1]
n_features = X_train.shape[1]

# model (Your implementation is correct)
class LogisticRegression (nn.Module) :
  def __init__(self, n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)

  def forward(self, X):
    x = self.linear(X)
    y_pred = torch.sigmoid(x)
    return y_pred

model = LogisticRegression(n_features)
loss = nn.BCELoss() # Binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# training
for epoch in range(100):
  y_pred = model(X_train)
  l = loss(y_pred, y_train)

  l.backward()

  optimizer.step()

  optimizer.zero_grad()

  if epoch % 10 == 0:
    print(f'epoch {epoch+1}: loss = {l:.3f}')

with torch.no_grad():
  y_pred = model(X_test)
  y_pred_class = y_pred.round()
  accuracy = y_pred_class.eq(y_test).sum() / float(y_test.shape[0]) #.e function basically if equal add + 1
  print(f'accuracy = {accuracy:.3f}')