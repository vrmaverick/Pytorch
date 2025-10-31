import torch
# if we want to calculate gradient wrt x we need requires_grad = True
# now on operation it will create a computaional greaph for us inputs + operator => outputs
# Backpropogation se gradient calculate



x = torch.randn(3, requires_grad = True)
print(x)

y = x+2
print(y)
# tensor([3.6090, 3.7639, 1.9134], grad_fn=<AddBackward0>)
z = y*y*2 # grad_fn=<MulBackward0>
print(z)
z = z.mean() #grad_fn=<MeanBackward0>) for making the value scalar 3 dim to 1 dim
print(z)

z.backward() #dz/dx  # works only on single value output or else set vector as attribute
print(x.grad) # gradients of tensor x

# gradients are calculated by jaccobian product with partial derivatives * gradient vector
# To prevent the vector from grad_fn-----> (removal) 3 options
x.requires_grad_(False)
y = x.detach()
with torch.no_grad():  y=x+1
print(x)

wt = torch.ones(4,requires_grad = True)
for epoch in range(3):
    model_output = (wt*3).sum()

    model_output.backward() #backward propogation but on iteration the valuse will be summed up

    print(wt.grad)
    wt.grad.zero_() # to reset the grads before use else it will be useless
    