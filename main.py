import torch
import matplotlib.pyplot as plt

L = 1.0
N = 100
M = 10
x = L*torch.linspace(L/M*0.5,L-L/M*0.5,M)
x.requires_grad = True
q = torch.ones(1,requires_grad=True)


def density(X,L,M,N,q):
    n = torch.zeros(M)
    xx = torch.linspace(0,L,M+1)
    xx.requires_grad = True
    dx = xx[1]-xx[0]
    # for x in X:
    #     i = int(x/dx)
    #     d = x- i*dx
    #     n[i]   += q*(1-d)
    #     if i+1 == xx.shape[0]:
    #        n[0]   += q*d
    #     else:
    #        n[i+1] += q*d

    i_dx = torch.divide(x,dx)
    i = i_dx.int()
    d = i_dx-i
    ii = i.tolist()
    n[ii] += (1-d)*q
    ii1 = i+1
    ii1 = torch.remainder(ii1,n.shape[0])
    ii1= ii1.tolist()
    n[ii1] += (d)*q
    return n
    # n[i]+=



    return n,xx
n,xx = density(x,L,M,N,1.0)

optimizer = torch.optim.SGD([x,q],lr=0.001)
loss = torch.sum(torch.abs(torch.ones_like(xx)-n))
loss.backward()
print(x.grad,q.grad)
optimizer.step()

qq = 0