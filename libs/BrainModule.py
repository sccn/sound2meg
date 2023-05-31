import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from collections import OrderedDict
import time

device = 'cuda'

class BrainModule(nn.Module):   
  def __init__(self, f_out, inchans, outchans, K, montage, n_subjects=None):
    super().__init__()
    self.D2 = 320
    self.outchans = outchans
    self.spatial_attention = SpatialAttention(inchans, outchans, K, montage[:,0], montage[:,1])
    self.conv = nn.Conv2d(outchans, outchans, 1, padding='same')
    if n_subjects:
      self.subject_layer = SubjectLayer(outchans, n_subjects)
    self.conv_blocks = nn.Sequential(*[self.generate_conv_block(k) for k in range(5)]) # 5 conv blocks
    self.final_convs = nn.Sequential(
      nn.Conv2d(self.D2, self.D2*2, 1),
      nn.GELU(),
      nn.Conv2d(self.D2*2, f_out, 1)
    )
    
  def generate_conv_block(self, k):
    kernel_size = (1,3)
    padding = 'same' # (p,0)
    return nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(self.outchans if k==0 else self.D2, self.D2, kernel_size, dilation=pow(2,(2*k)%5), padding=padding)),
      ('bn1',   nn.BatchNorm2d(self.D2)), 
      ('gelu1', nn.GELU()),
      ('conv2', nn.Conv2d(self.D2, self.D2, kernel_size, dilation=pow(2,(2*k+1)%5), padding=padding)),
      ('bn2',   nn.BatchNorm2d(self.D2)),
      ('gelu2', nn.GELU()),
      ('conv3', nn.Conv2d(self.D2, self.D2*2, kernel_size, padding=padding)),
      ('glu',   nn.GLU(dim=1))
    ]))

  def forward(self, x, subj_indices=None):
    x = self.spatial_attention(x).unsqueeze(2) # add dummy dimension at the end
    x = self.conv(x)
    x = self.subject_layer(x, subj_indices)
        
    for k in range(len(self.conv_blocks)):
      if k == 0:
        x = self.conv_blocks[k](x)
      else:
        x_copy = x
        for name, module in self.conv_blocks[k].named_modules():
          if name == 'conv2' or name == 'conv3':
            x = x_copy + x # residual skip connection for the first two convs
            x_copy = x.clone() # is it deep copy?
          x = module(x)
    x = self.final_convs(x)
        
    return x.squeeze(2)
    
class SubjectLayer(nn.Module):
  def __init__(self, outchans, n_subjects):
    super().__init__()
    self.subj_layers = nn.Sequential(*[nn.Conv2d(outchans, outchans, 1, padding='same') for i in range(n_subjects)])

  def forward(self, x, subj_indices):
    for i in range(x.shape[0]):
      x[i] = self.subj_layers[subj_indices[i]](x[i].clone())
    return x
        
class SpatialAttention(nn.Module):
  def __init__(self,in_channels, out_channels, K, x, y):
    super().__init__()
    self.outchans = out_channels
    self.inchans = in_channels
    self.K = K
    self.x = x.to(device=device)
    self.y = y.to(device=device)
    self.compute_cos_sin()           
    # trainable parameter:
    self.z = Parameter(torch.randn(self.outchans, K*K, dtype = torch.cfloat,device=device)/(32*32)) # each output channel has its own KxK z matrix
    self.z.requires_grad = True
            
  def compute_cos_sin(self):
    kk = torch.arange(1, self.K+1, device=device)
    ll = torch.arange(1, self.K+1, device=device)
    cos_fun = lambda k, l, x, y: torch.cos(2*torch.pi*(k*x + l*y))
    sin_fun = lambda k, l, x, y: torch.sin(2*torch.pi*(k*x + l*y))
    self.cos_matrix = torch.stack([cos_fun(kk[None,:], ll[:,None], x, y) for x, y in zip(self.x, self.y)]).reshape(self.inchans,-1).float()
    self.sin_matrix = torch.stack([sin_fun(kk[None,:], ll[:,None], x, y) for x, y in zip(self.x, self.y)]).reshape(self.inchans,-1).float()

  def forward(self, X):            
    a = torch.matmul(self.z.real, self.cos_matrix.T) + torch.matmul(self.z.imag, self.sin_matrix.T)
    # Question: divide this with square root of KxK? to stablize gradient as with self-attention?
    a = F.softmax(a, dim=1) # softmax over all input chan location for each output chan
                                            # outchans x  inchans
                
            # X: N x 273 x 360            
    X = torch.matmul(a, X) # N x outchans x 360 (time)
                                   # matmul dim expansion logic: https://pytorch.org/docs/stable/generated/torch.matmul.html
    return X
