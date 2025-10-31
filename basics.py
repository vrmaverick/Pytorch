import torch
x = torch.empty(3,3,4) # shape, #Dtype = torch.float16 parameter
# torch.zeros
#torch.rand
#torch.randint(5,(3,3))
# torch.ones
# x.dtype : float 32
print(x)

x = torch.tensor([1,2,3,4,5])
print(x)

x = torch.randint(3,4,(3,3))
y = torch.randint(3,4,(3,3))

print(x+y)
y.add_(x) # yeh dash jo hot hai function ke baad it is inplace operation
print(y)

print(y[1,1].item()) #use only when you have slected 1 element in tensor
y = x.view(9) #4x4 =16
z = x.view(-1,1) #-1 daalne pepytorch will determine the remaining dimensions by its own keepig 2