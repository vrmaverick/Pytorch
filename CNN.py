import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as f

num_epochs = 4
batch_size = 128
lr = 0.001

# Transform

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True , download = True, transform= transform)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False , download = True, transform= transform)
train_data_loader = torch.utils.data.DataLoader (train_dataset, shuffle = True, batch_size = batch_size)
test_data_loader = torch.utils.data.DataLoader (test_dataset, shuffle = False, batch_size = batch_size)

classes = train_dataset.classes
print("CIFAR-10 Classes:", classes)

image, label = train_dataset[0]
print(image.shape)

class Convonet(nn.Module):
  def __init__(self):
    super(Convonet,self).__init__()
    self.conv_1 = nn.Conv2d(3,16,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv_2 = nn.Conv2d(16,32,5)

    # calculating final input size for the fc1
    # calculate by formula or just do output channel *5*5
    self.fc1 = nn.Linear(32*5*5,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,32)
    self.fc4 = nn.Linear(32,10)


  def forward(self,input) :
    X = self.pool(f.relu(self.conv_1(input)))
    X = self.pool(f.relu(self.conv_2(X))) # Pass the output of the previous layer (X)
    # Flattening after convolution layer

    X = X.view(-1,32*5*5) # Use X.view() instead of self.view()
    X = f.relu(self.fc1(X))
    X = f.relu(self.fc2(X))
    X = f.relu(self.fc3(X))
    output = self.fc4(X)

    return output

model = Convonet()

Criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.SGD(model.parameters(),lr = lr)
n_total_steps = len(train_data_loader)
loss_history = []


for epoch in range(num_epochs):
  for i,(image,labels) in enumerate(train_data_loader):
    output = model(image)
    loss = Criterion(output,labels)
    loss_history.append(loss.item())
    # Backpropogation but we reset the gradients
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step() # Update weights

    if i%10 == 0:
      print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
print("Training Done")

plot_loss = []
for i in range(len(loss_history)):
  if i%391 == 0:
    plot_loss.append(loss_history[i])
plt.figure(figsize=(10, 6)) # Create a new figure
plt.plot(plot_loss) # Plot the list of loss values
plt.title('Training Loss Over Steps') # Set the title
plt.xlabel('Training Step') # Label the x-axis
plt.ylabel('Loss Value') # Label the y-axis
plt.grid(True) # Add a grid for better readability
plt.show()