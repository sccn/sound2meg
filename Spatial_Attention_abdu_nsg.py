# -*- coding: utf-8 -*-
"""Spatial_Attention.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/sccn/sound2meg/blob/main/Spatial_Attention.ipynb
"""

import torch
import os
from torch.autograd import Variable
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from scipy.io import loadmat
from dataset_loading import Sound2MEGDataset
from torch.utils.data import Dataset, DataLoader, random_split
import gc
import sys
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.tracebacklimit = 0

class SubjectLayer(nn.Module):
  def __init__(self):
    super(SubjectLayer, self).__init__()
    self.layers = []

    for i in range(124): #124 subjects
      layer = nn.Conv2d(270, 270, 1).to(device)
      self.layers.append(layer)
      
  def forward(self, x, s_idx):
    x = x.unsqueeze(1)
    for i in range(len(x)):
      x[i] = self.layers[s_idx[i]](x[i].clone())
    return x[:, 0, :, :]

class SpatialAttention(nn.Module):
  def __init__(self,in_channels, out_channels, K, path):
    super(SpatialAttention, self).__init__()
    self.out = out_channels
    self.input = in_channels
    self.K = K
    self.z = Parameter(torch.randn(self.out, K*K, dtype = torch.cfloat)/(32*32))
    self.z.requires_grad = True
    self.positions = loadmat(path + 'electrode_positions.mat')
    self.positions = self.positions['positions']
    self.x = torch.tensor(self.positions[:, 0]).to(device)
    self.y = torch.tensor(self.positions[:, 1]).to(device)
    self.cos_v = []
    self.sin_v = []
    self.cos = []
    self.sin = []
    for i in range(in_channels):
      self.cos_v = []
      self.sin_v = []
      for k in range(K):
        for l in range(K):
          self.cos_v.append(torch.cos(2*math.pi*(k*self.x[i]+l*self.y[i])))
          self.sin_v.append(torch.sin(2*math.pi*(k*self.x[i]+l*self.y[i])))
      self.cos.append(torch.stack(self.cos_v))
      self.sin.append(torch.stack(self.sin_v))
    self.cos = torch.stack(self.cos).to(device)
    self.sin = torch.stack(self.sin).to(device)
  def forward(self, X):
    N = X.size()[0]
    SA = torch.zeros(N, 270, 360).to(device)
    z_r = self.z.real
    z_i = self.z.imag
    a = (torch.mm(z_r.float(), torch.transpose(self.cos, 0, 1).float()) + torch.mm(z_i.float(), torch.transpose(self.sin, 0, 1).float())).to(device)
    exp2 = torch.sum(torch.exp(a[:, 0:self.out]), 1).to(device)
    exp2 = torch.transpose(exp2.unsqueeze(0), 0, 1)
    exp2 = torch.mm(exp2, torch.ones(1, 360).to(device))
    for i in range(N):
      exp1 = torch.mm(torch.exp(a), X[i]).to(device)
      SA[i] = (exp1/exp2).to(device)
      #SA[i] = SpatialAttentionSoftmax(self.input, self.out, X[i], a)
    return SA

class Net(nn.Module):
  def __init__(self, path, F):
    super(Net, self).__init__()
    self.SA = SpatialAttention(273, 270, 32, path)
    self.Subject = SubjectLayer().to(device)
    self.F = F
    self.conv1 = nn.Conv2d(270, 270, (1, 1)).to(device)
    self.conv2 = nn.Conv2d(320, 640, (1, 1)).to(device)
    self.conv3 = nn.Conv2d(640, self.F, (1, 1)).to(device)
    self.gelu = nn.GELU().to(device)
    self.loop_convs = []
    self.loop_batchnorms = []
    self.loop_gelus = []
    self.loop_glus = []
    for k in range(1, 6):
      p = pow(2,(2*k)%5)
      q = pow(2,(2*k+1)%5)
      self.convs = []
      self.batchnorms = []
      self.gelus = []
      self.convs.append(nn.Conv2d(320, 320, (3, 1), dilation = p, padding = (p, 0)).to(device))
      self.convs.append(nn.Conv2d(320, 320, (3, 1), dilation = q, padding = (q, 0)).to(device))
      self.convs.append(nn.Conv2d(320, 640, (3, 1), dilation = 2, padding = (2, 0)).to(device))
      for i in range(2):
        self.batchnorms.append(nn.BatchNorm2d(320).to(device))
        self.gelus.append(nn.GELU().to(device))
      self.loop_convs.append(self.convs)
      self.loop_batchnorms.append(self.batchnorms)
      self.loop_gelus.append(self.gelus)
      self.loop_glus.append(nn.GLU().to(device))
    self.loop_convs[0][0] = nn.Conv2d(270, 320, (3, 1), dilation = 1, padding = (1, 0)).to(device)
  def forward(self, x, s_idx):
    x = self.SA(x).unsqueeze(3)
    x = self.conv1(x)
    x = self.Subject(x, s_idx)
    for i in range(5):
      if i == 0:
        x = self.loop_convs[0][0](x)
        x = self.loop_batchnorms[0][0](x)
        x = self.loop_gelus[0][0](x)
        x = self.loop_convs[0][1](x)
        x = self.loop_batchnorms[0][1](x)
        x = self.loop_gelus[0][1](x)
        x = self.loop_convs[0][2](x)
        x = torch.transpose(x, 3, 1)
        x = self.loop_glus[0](x)
        x = torch.transpose(x, 3, 1)
      else:
        x1 = self.loop_convs[i][0](x)
        x1 = self.loop_batchnorms[i][0](x)
        x1 = self.loop_gelus[i][0](x)
        x2 = x + x1
        x3 = self.loop_convs[i][1](x2)
        x3 = self.loop_batchnorms[i][1](x3)
        x3 = self.loop_gelus[i][1](x3)
        x4 = x2 + x3
        x5 = self.loop_convs[i][2](x4)
        x5 = torch.transpose(x5, 3, 1)
        x5 = self.loop_glus[i](x5)
        x = torch.transpose(x5, 3, 1)
    x_out = self.conv2(x)
    x_out = self.gelu(x_out)
    x_out = self.conv3(x_out)
    return x_out

def CLIP_loss(Z, Y):
    '''
    New loss using cross entropy implementation
    '''
    N = Y.size(dim = 0) # batch size
    Z_row = torch.reshape(Z, (N, -1)) # flatten to be N x F
    Y_row = torch.reshape(Y, (N, -1)) # flatten to be N x F
    inner_product = (torch.mm(Z_row, Y_row.T)/(N*N)).to(device) # N x N. The normalization?

    target = torch.arange(N, device=device)
    loss_brain = torch.nn.functional.cross_entropy(inner_product, target)
    loss_sound = torch.nn.functional.cross_entropy(inner_product.T, target)
    loss = ((loss_brain + loss_sound)/2).to(device)
    
    return loss

embedding_type = 'Wav2Vec'
F = 120
dataset = Sound2MEGDataset('/expanse/projects/nsg/external_users/public/arno/', embedding_type)
training_data, validation_data, test_data = random_split(dataset, [11497, 3285, 1642], generator=torch.Generator().manual_seed(42))
Training_Data_Batches = DataLoader(training_data, batch_size = 128, shuffle = True)
Validation_Data_Batches = DataLoader(validation_data, batch_size = 128, shuffle = True)
BrainModule = Net('/expanse/projects/nsg/external_users/public/arno/', F)
BrainModule.to(device)
optimizer = optim.Adam(BrainModule.parameters(), lr = 0.0003)
loss_train = []
loss_val = []

for i in range(50):
  loss_t = 0
  loss_v = 0
  BrainModule.train()
  for MEG, WAV, Sub in Training_Data_Batches:
    Sub = Sub.tolist()
    optimizer.zero_grad()
    Z = BrainModule(MEG.to(device), Sub)
    Z = Z[:, :, :, 0]
    loss = CLIP_loss(Z.float(), WAV.abs().float().to(device))
    print(loss.item())
    loss.backward()
    loss_t = loss_t + loss.item()
    optimizer.step()
  loss_train.append(loss_t/(len(Training_Data_Batches)))
  BrainModule.eval()
  for MEG_val, WAV_val, Sub_val in Validation_Data_Batches:
    with torch.no_grad():
      Z_val = BrainModule(MEG_val.to(device), Sub_val)
      Z_val = Z_val[:, :, :, 0]
      loss = CLIP_loss(Z_val.float(), WAV_val.abs().float().to(device))
    loss_v = loss_v + loss.item()
  loss_val.append(loss_v/len(Validation_Data_Batches))
  gc.collect()
  torch.cuda.empty_cache()

print(loss_train)
print(loss_val)
