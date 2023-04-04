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
from sklearn.preprocessing import RobustScaler
import gc
import sys
import numpy as np
import glob
import csv

filePath = '/content/drive/MyDrive/sound2meg/'
filePath = '/expanse/projects/nsg/external_users/public/arno/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.tracebacklimit = 0

class Sound2MEGDataset(Dataset):
  def __init__(self, path):
    #self.wav_files = wav_files
    self.path = path
    self.sizes = np.empty(0, dtype=int)
    self.subjects = []
    for subject in range(0, 126):
        if os.path.exists(filePath + 'MEG_Signals/S%03dT000.npy'%subject):
            files = glob.glob(filePath + 'MEG_Signals/S%03dT*.npy'%subject)
            self.sizes = np.append(self.sizes, len(files))
            self.subjects.append( '%03d'%subject )
                                
    self.filename = []
#     for file in listdir(path + 'mat_files'):
#       self.subjects.append(file[6:9])             #Making a list of all the mat files
  def __len__(self):
    return sum(self.sizes)
  def __getitem__(self, idx):
    for i in range(len(self.sizes)):              #Taking a cummulative sum of all the sizes
      if np.cumsum(self.sizes)[i] > idx:          #This way we will know the ranges of the different MEG Signal samples in the context of the entire dataset
        idx_file = i                              #For example, say the sizes of the mat files are [3, 5, 2]
        break                                     #Using the cummulative function, we get [3, 8, 10]
    #mat_file = loadmat(self.path + 'mat_files/' + self.filename[idx_file]) #This tells that samples 0 to 2 are in the first file, 3 to 7 in the second and so on...
    if idx_file == 0:                             #Accordingly, we see which sample is in which range by finding the smallest value in the cumsum larger than idx                           
      idx_sound = idx
    else:
      idx_sound = idx - np.cumsum(self.sizes)[idx_file - 1]
    
    fileName = self.path + 'MEG_Signals/S'+self.subjects[idx_file]+'T%03d.npy'%idx_sound
    MEG_Signal = np.load(fileName)
    with open(self.path + 'MEG_Signals/audios.csv', 'r') as f:
      audios = list(csv.reader(f))
      audio_file = audios[idx_file][idx_sound][2:5]
    #Finding the associated audiofile, and extracting only the number of the file out of it
    #audio_file = mat_file['audiofiles'][0, idx_sound][0][0:3]
    #The filenames are in the format 'mel_<number>.npy' so importing accordingly
    Sound_Signal = np.load(self.path + 'Mel_Embedding120/mel_' + audio_file + '.npy')
    #mat_file = mat_file['data']
    MEG_Signal = np.float32(MEG_Signal[:273, :])
    #Baseline Correction
    mean_5 = np.mean(MEG_Signal[:,:60], axis=1)
    MEG_Signal = MEG_Signal - mean_5[:,None]
    #Robust Scaler
    scaler = RobustScaler().fit(MEG_Signal)
    MEG_Signal = scaler.transform(MEG_Signal)
    return torch.from_numpy(MEG_Signal), torch.from_numpy(Sound_Signal) , idx_file

class SubjectLayer(nn.Module):
  def __init__(self):
    super(SubjectLayer, self).__init__()
    self.layers = []

    for i in range(124): #124 subjects
      layer = nn.Conv2d(270, 270, 1)
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
    SA = torch.zeros(N, 270, 360)
    z_r = self.z.real
    z_i = self.z.imag
    a = (torch.mm(z_r.float(), torch.transpose(self.cos, 0, 1).float()) + torch.mm(z_i.float(), torch.transpose(self.sin, 0, 1).float())).to(device)
    exp2 = torch.sum(torch.exp(a[:, 0:self.out]), 1).to(device)
    exp2 = torch.transpose(exp2.unsqueeze(0), 0, 1)
    exp2 = torch.mm(exp2, torch.ones(1, 360).to(device))
    for i in range(N):
      exp1 = torch.mm(torch.exp(a), X[i]).to(device)
      SA[i] = exp1/exp2
      #SA[i] = SpatialAttentionSoftmax(self.input, self.out, X[i], a)
    return SA

class Net(nn.Module):
  def __init__(self, path, F):
    super(Net, self).__init__()
    self.SA = SpatialAttention(273, 270, 32, path)
    self.Subject = SubjectLayer()
    self.F = F
  def forward(self, y, s_idx):
    x1 = self.SA(y).unsqueeze(0)
    x2 = x1.permute((1, 2, 3, 0)) # subject attention?
    x3 = nn.Conv2d(270, 270, (1, 1))(x2)
    x = self.Subject(x3, s_idx)
    for k in range(1,6):
      p = pow(2,(2*k)%5)
      q = pow(2,(2*k+1)%5)
      if k == 1:
        x = nn.Conv2d(270, 320, (3, 1), dilation = 1, padding = (1, 0))(x)
        x = nn.BatchNorm2d(320)(x)
        x = nn.GELU()(x)
        x = nn.Conv2d(320, 320, (3, 1), dilation = 1, padding = (1, 0))(x)
        x = nn.BatchNorm2d(320)(x)
        x = nn.GELU()(x)
        x = nn.Conv2d(320, 640, (3, 1), dilation = 2, padding = (2, 0))(x)
        x = torch.transpose(x, 3, 1)
        x = nn.GLU()(x)
        x = torch.transpose(x, 3, 1)
      else:
        x1 = nn.Conv2d(320, 320, (3, 1), dilation = p, padding = (p, 0))(x)
        x1 = nn.BatchNorm2d(320)(x1)
        x1 = nn.GELU()(x1)
        x2 = x + x1
        x3 = nn.Conv2d(320, 320, (3, 1), dilation = q, padding = (q, 0))(x2)
        x3 = nn.BatchNorm2d(320)(x3)
        x3 = nn.GELU()(x3)
        x4 = x2 + x2
        x_out = nn.Conv2d(320, 640, (3, 1), dilation = 2, padding = (2, 0))(x4)
        x_out = torch.transpose(x_out, 3, 1)
        x_out = nn.GLU()(x_out)
        x_out = torch.transpose(x_out, 3, 1)
    x_out = nn.Conv2d(320, 640, (1, 1))(x_out)
    x_out = nn.GELU()(x_out)
    x_out = nn.Conv2d(640, self.F, (1, 1))(x_out)
    return x_out

def CLIP_loss(Z, Y):
  N = Y.size(dim = 0)
  #inner_product = torch.zeros(N, N)
  log_softmax = torch.zeros(N).to(device)
  Z_row = torch.reshape(Z, (N, -1)).to(device)
  Y_row = torch.reshape(Y, (N, -1)).to(device)
  inner_product = (torch.mm(Z_row, torch.transpose(Y_row, 1, 0))/(N*N)).to(device)
  for i in range(N):
    inn = inner_product[i, :].to(device)
    log_softmax[i] = torch.log(nn.functional.softmax(inn, -1))[i]
  return sum(-1*log_softmax)

print('Starting...')
sys.stdout.flush()
Dataset = Sound2MEGDataset('/expanse/projects/nsg/external_users/public/arno/')
training_data, validation_data, test_data = random_split(Dataset, [11497, 3285, 1642], generator=torch.Generator().manual_seed(42))
Training_Data_Batches = DataLoader(training_data, batch_size = 128, shuffle = True)
Validation_Data_Batches = DataLoader(validation_data, batch_size = 128, shuffle = True)
BrainModule = Net('/expanse/projects/nsg/external_users/public/arno/', 120)
BrainModule.to(device)
optimizer = optim.Adam(BrainModule.parameters(), lr = 0.0003)
loss_train = []
loss_val = []

for i in range(80):
  loss_t = 0
  loss_v = 0
  for j in range(13):
    for MEG, WAV, Sub in Training_Data_Batches:
      Sub = Sub.tolist()
      Z = BrainModule(MEG.to(device), Sub)
      Z = Z[:, :, :, 0]
      loss = CLIP_loss(Z.float(), WAV.abs().float().to(device))
      torch.autograd.set_detect_anomaly(True)
      optimizer.zero_grad()
      loss.backward()
      loss_t = loss_t + loss.item()
      optimizer.step()
  print("Average training loss: " + str(loss_t/len(Training_Data_Batches)), end='')
  sys.stdout.flush()
  loss_train.append(loss_t/(13*len(Training_Data_Batches)))
  for MEG_val, WAV_val, Sub_val in Validation_Data_Batches:
    Z_val = BrainModule(MEG_val.to(device), Sub_val)
    loss = CLIP_loss(Z_val.float(), WAV_val.abs().float().to(device))
    # print("Batch validation loss: " + str(loss.item()))
    loss_v = loss_v + loss.item()
  print("Average validation loss: " + str(loss_v/len(Validation_Data_Batches)))
  sys.stdout.flush()
  loss_val.append(loss_v/len(Validation_Data_Batches))
  gc.collect()
  torch.cuda.empty_cache()

print(loss_train)
print(loss_val)
