import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# MNIST

input_size = 784 #(28X28) jisse 1d banate hai
hidden_size = 500
num_classes = 10
epochs = 10
batch_size =64
lr = 0.001

train_dataset = torchvision.datasets.MNIST(root = './content/drive/MyDrive/datasets',train = True,transform = transforms.ToTensor(),download = True)
test_dataset = torchvision.datasets.MNIST(root = './content/drive/MyDrive/datasets',train = False,transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = False)

examples = iter(train_loader)
samples,labels = next(examples)
print(samples.shape,labels.shape)

for i in range(10) :
    plt.subplot(2, 5, i+1)           # (rows, cols, index)
    plt.imshow(samples[i][0], cmap='gray')  # show the 1-channel image
    plt.title(f"Label: {labels[i].item()}") # show its label above image
    plt.axis('off')
plt.show()

class Feedforwardimages(nn.Module):
  def __init__(self,input_size,hidden_size,num_classes):
    super(Feedforwardimages,self).__init__()
    self.flatten = nn.Flatten()
    self.l1 = nn.Linear(input_size,hidden_size)
    self.relu = nn.ReLU() # activation function
    self.l2 = nn.Linear(hidden_size,num_classes)

  def forward(self,input) :
    X = self.flatten(input)
    X = self.l1(X)
    X = self.relu(X)
    output= self.l2(X)
    return output

model = Feedforwardimages(input_size,hidden_size,num_classes)

# Loss and optimizer
criterion= nn.CrossEntropyLoss() # Applies softmax at the end for us so no need to define in the flow
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_total_steps = len(train_loader)
for e in range(epochs) :
  for i , (images,labels) in enumerate(train_loader) :
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    #Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%64 == 0:
      print(f'{e+1}/{epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')


with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images,labels in test_loader:
    outputs = model(images)
    #value and index (Class Label) :
    _,predicted = torch.max(outputs,1) # along dimension 1
    n_samples += labels.shape[0]
    n_correct += (predicted == labels).sum().item()

  accuracy = 100 * n_correct / n_samples
  print(f'Accuracy = {accuracy}')

  torch.save(model.state_dict(), './content/drive/MyDrive/datasets/pytorchmodel.pth') # only saves weights not architecture which is reccomended
# To load
model = Feedforwardimages(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('./content/drive/MyDrive/datasets/pytorchmodel.pth'))
model.eval()  # put model in evaluation mode (turns off dropout, batchnorm, etc.)
