import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def forward(self, input, input_shape):
        return input.view(input.size(0), input_shape)

def append_file(iterable, path, sep='\t', num=None):
    with open(path, 'a+') as f:
        if num is not None:
            f.write(str(num) + sep)
        f.write(sep.join(iterable) + '\n')
def write_file(content, path):
    with open(path, 'w') as f:
        f.write(content)
        f.write('\n')

def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(data)

    batch_count = 0
    data_size = len(data)
    while batch_count * batch_size + batch_size <= data_size:
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield batch_count, data[start:end]

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torchvision.utils.save_image(tensor, filename, nrow, padding, normalize, range, scale_each, pad_value)

def sample_gumbel_softmax(logits, temperature, eps=1e-20):
    unif = torch.rand(logits.shape).cuda()
    gumbel = - torch.log(-torch.log(unif + eps) + eps)
    return F.softmax( (logits + gumbel) / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    z = sample_gumbel_softmax(logits, temperature)
    shape = z.size()
    _, ind = z.max(dim=-1)
    z_hard = torch.zeros_like(z).view(-1, shape[-1])
    z_hard.scatter_(1, ind.view(-1, 1), 1)
    z_hard = z_hard.view(*shape)
    z_hard = (z_hard - z).detach() + z
    return z_hard

def activation_func(activation=None):
    # TODO: 检查激活函数是否存在于库
    if activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'relu':
        return torch.relu
    elif activation == 'tanh':
        return torch.tanh
    else:
        return lambda x: x