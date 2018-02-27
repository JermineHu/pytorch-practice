import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

def make_features(x):
    """Builds features i.e. a matrix with colums [x,x^2,x^3]. """
    x=x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)

w_target=torch.FloatTensor([0.5,3,2.4]).unsqueeze(1)
b_target=torch.FloatTensor([0.9])

def f(x):
    """ Approximated funhction."""
    return x.mm(w_target)+b_target[0]

def get_batch(batch_size=32):
    """Build a batch i.e. (x,f(x)) pair."""
    random=torch.randn(batch_size)
    x=make_features(random)
    y=f(x)

    if torch.cuda.is_available():
        return Variable(x).cuda(),Variable(y).cuda()
    else:
        return Variable(x),Variable(y)

#Define model
class PolyModel(nn.Module):
    def __init__(self):
        super(PolyModel, self).__init__()
        self.poly=nn.Linear(3,1)
    def forward(self,x):
        return self.poly(x)

if torch.cuda.is_available():
    model=PolyModel().cuda()
else:
    model=PolyModel()

criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=1e-3)

epoch=0

while True:
    #Get data
    batch_x,batch_y=get_batch()
    # Forward pass
    output=model(batch_x)
    loss=criterion(output,batch_y)
    print_loss=loss.data[0]
    # Reset gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    epoch+=1
    if print_loss<1e-3:
        break

print("----->",)

