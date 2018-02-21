import torch
import numpy as np
# def a Longtensor
a= torch.LongTensor([[2,3],[4,8],[7,9]])

print("a is :{}".format(a))
print("a size is {}".format(a.size()))

b= torch.LongTensor([[2,3],[4,8],[7,9]])

print("b is :{}".format(b))
print("b size is {}".format(b.size()))

c=torch.zeros((3,2))

print('zero tensor:{}'.format(c))

d=torch.randn((3,2))

print("normal randon is :{}".format(d))

print("a is :{}".format(a))

a[0,1]=100

print("a changed value is :{}".format(a))

nummpy_b=a.numpy()

print('convert to numpy is \n {}'.format(nummpy_b))

e=np.array([[2,3],[4,5]])
torch_e=torch.from_numpy(e)

print("from numpy to torch.Tensor is {}".format(torch_e))

f_torch_e=torch_e.float()

print('change data type tp float tensor:{}'.format(f_torch_e))

if torch.cuda.is_available():  # Judge the pc whether support cuda.
    a_cuda=a.cuda()
    print(a_cuda)

