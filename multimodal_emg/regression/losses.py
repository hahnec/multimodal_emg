import numpy
import torch


l1_norm = lambda y, y_star: abs(y-y_star)
l2_norm = lambda y, y_star: (y-y_star)**2

def huber_loss(y, y_star, delta=1):

    a = l2_norm(y, y_star)

    loss = torch.zeros_like(a) if isinstance(a, torch.Tensor) else numpy.zeros_like(a)
    idcs = a < delta
    loss[idcs] = (a[idcs]**2)/2
    loss[~idcs] = delta*(abs(a[~idcs]) - delta/2)

    return loss
