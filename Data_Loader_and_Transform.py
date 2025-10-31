import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset # <-- This is the key fix for the NameError

class LoadDataset(Dataset):
  def __init__(self):
    # data Loading
    # Ensure this path is correct for your environment:
    csv_path = '/content/drive/MyDrive/datasets/wine.csv'
    xy = np.loadtxt(csv_path, delimiter=',', dtype=np.float32, skiprows=1)

    # Features (all columns except the first one)
    self.x = torch.from_numpy(xy[:, 1:])

    # Labels (only the first column)
    self.y = torch.from_numpy(xy[:, [0]])

    # Total number of samples
    self.n_samples = xy.shape[0]

  def __getitem__(self, index):
    # This allows you to access a sample with dataset[index]
    return self.x[index], self.y[index]

  def __len__(self):
    # This allows you to get the size of the dataset with len(dataset)
    return self.n_samples

# Instantiating the dataset
dataset = LoadDataset()

first_data = dataset[0]
features, labels = first_data
print(first_data)
print(f'feature = {features}, Label =  {labels}')

class winedata(Dataset):
  def __init__(self,transform = None):
    # data Loading
    # Ensure this path is correct for your environment:
    csv_path = '/content/drive/MyDrive/datasets/wine.csv'

    xy = np.loadtxt(csv_path,delimiter = ',',dtype = np.float32,skiprows=1)
    self.x = xy[:,1:]
    self.y = xy[:,[0]]
    print(self.x.shape)
    self.n_samples = xy.shape[0]
    # print(self.x,self.y)
    self.transform = transform

  def __getitem__(self,index):
    sample = self.x[index],self.y[index]
    if self.transform:
      sample = self.transform(sample)
    return sample

  def __len__(self):
    return self.n_samples

class ToTensor: # basicallt its a transform class called below
  def __call__(self,sample):
    inputs,labels = sample
    return torch.from_numpy(inputs),torch.from_numpy(labels)

class MulTransform:
  def __init__(self,factor):
    self.factor = factor

  def __call__(self,sample):
    inputs,labels = sample
    inputs *= self.factor
    return inputs,labels

dataset = winedata(transform = None)
print(dataset[0]) #We get Array

dataset = winedata(transform = ToTensor())
print(dataset[0]) #We get Array

composed = torchvision.transforms.Compose([ToTensor(),MulTransform(2)])
dataset = winedata(transform = composed)
print(dataset[0])



# first_data = dataset[0]
# features, labels = first_data
# print(first_data)
# print(f'feature = {features}, Label =  {labels}')