import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import os
import sys
import datetime
import csv
import pandas as pd
from collections import OrderedDict

import torchvision
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

# For visualize input
from torch.utils.tensorboard import SummaryWriter
import io
import torchvision
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):   
  def __init__(self, F_out, inchans, outchans, K):
    super().__init__()
    self.D2 = 320
    self.outchans = outchans
    #self.spatial_attention = SpatialAttention(inchans, outchans, K, montage[:,0], montage[:,1])
    self.conv = nn.Conv2d(outchans, outchans, 1, padding='same')
    self.conv_blocks = nn.Sequential(*[self.generate_conv_block(k) for k in range(5)]) # 5 conv blocks
    self.final_convs = nn.Sequential(
      nn.Conv2d(self.D2, self.D2*2, 1),
      nn.GELU(),
      nn.Conv2d(self.D2*2, F_out, 1)
    )
    self.l1 = nn.Linear(256*F_out, 2)
    
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

  def forward(self, x):
    x = x[:,0].unsqueeze(3) # add dummy dimension at the end
    x = self.conv(x)
        
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
    x = torch.flatten(x, 1)
    x = F.softmax(self.l1(x), -1)
        
    return x    
        
class SpatialAttention(nn.Module):
  def __init__(self,in_channels, out_channels, K, x, y):
    super().__init__()
    self.outchans = out_channels
    self.inchans = in_channels
    self.K = K
    self.x = x.to(device=device)
    self.y = y.to(device=device)
    self.x_drop = random.uniform(0, 1)
    self.y_drop = random.uniform(0, 1)
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
    for i in range(self.inchans):
      distance = (self.x_drop - self.x[i])**2 + (self.y_drop - self.y[i])**2
      if distance < 0.1:
        a[:, i] = 0
        
    a = F.softmax(a, dim=1) # softmax over all input chan location for each output chan
                                            # outchans x  inchans
                
            # X: N x 273 x 360            
    X = torch.matmul(a, X) # N x outchans x 360 (time)
                                   # matmul dim expansion logic: https://pytorch.org/docs/stable/generated/torch.matmul.html
    return X


class EEGDataset(Dataset):
    def __init__(self, x, y, train, val):
        super(EEGDataset).__init__()
        assert x.shape[0] == y.size
        self.x = x
        self.y = [y[0][i] for i in range(y.size)]
        self.train = train
        self.val = val

    def __getitem__(self,key):
        return (self.x[key], self.y[key])

    def __len__(self):
        return len(self.y)
        
def load_data(path, role, winLength, numChan, srate, feature, one_channel=False, version=""):
    """
    Load dataset
    :param  
        path: Filepath to the dataset
        role: Role of the dataset. Can be "train", "val", or "test"
        winLength: Length of time window. Can be 2 or 15
        numChan: Number of channels. Can be 24 or 128
        srate: Sampling rate. Supporting 126Hz
        feature: Input feature. Can be "raw", "spectral", or "topo"
        one_channel: Where input has 1 or 3 channel in depth dimension. Matters when load topo data as number of input channels 
                are different from original's
        version: Any additional information of the datafile. Will be appended to the file name at the end
    """
    transform = T.Compose([
        T.ToTensor()
    ])
    if version:
        f = pd.read_pickle(path + f"child_mind_x_{role}_{winLength}s_{numChan}chan_{feature}_{version}.pkl")
    else:
        f = pd.read_pickle(path + f"child_mind_x_{role}_{winLength}s_{numChan}chan_{feature}.pkl")
    x = f[f'X_{role}']
    if feature == 'raw':
        x = np.transpose(x,(0,2,1))
        x = np.reshape(x,(-1,1,numChan,winLength*srate))
    elif feature == 'topo':
        if one_channel:
            samples = []
            for i in range(x.shape[0]):
                image = x[i]
                b, g, r = image[0,:, :], image[1,:, :], image[2,:, :]
                concat = np.concatenate((b,g,r), axis=1)
                samples.append(concat)
            x = np.stack(samples)
            x = np.reshape(x,(-1,1,x.shape[1],x.shape[2]))
    
    if version:
        f = pd.read_pickle(path + f"child_mind_y_{role}_{winLength}s_{numChan}chan_{feature}_{version}.pkl")
    else:
        f = pd.read_pickle(path + f"child_mind_y_{role}_{winLength}s_{numChan}chan_{feature}.pkl")
    y = f[f'Y_{role}']
   
    return EEGDataset(x, y, role=='train', role=='val')



def create_model():
    return Net(F_out = 120, inchans = 24, outchans = 24, K = 32)


def check_accuracy(loader, model, device, dtype):
    '''
    Check accuracy of the model 
    param:
        loader: An EEGDataset object
        model: A PyTorch Module to test
        device: cpu or cuda
        dtype: value type
        logger: Logger object for logging purpose
    '''
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc

def train(model, optimizer, epochs):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - logger: Logger object for logging purpose
    Returns: Nothing, but prints model accuracies during training.
    """
    path = '/expanse/projects/nsg/external_users/public/arno/child_mind_abdu/'
    winLength = 2
    numChan = 24
    srate = 128
    feature = 'raw'
    one_channel = False
    role = 'train'
    train_data = load_data(path, role, winLength, numChan, srate, feature, one_channel)

    role = 'val'
    val_data = load_data(path, role, winLength, numChan, srate, feature, one_channel)

    batch_size = 70 
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=batch_size)

    dtype = torch.float32
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    loss_t = []
    for e in range(epochs):
        loss_train = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)
            loss_train = loss_train + loss

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
        loss_t.append(loss_train)

        train_acc = check_accuracy(loader_train, model, device, dtype)
        print('Train Accuracy at Epoch ' + str(e) + ': ' + str(train_acc)) 
        val_acc = check_accuracy(loader_val, model, device, dtype)
        print('Val Accuracy at Epoch ' + str(e) + ': ' + str(val_acc))

    print(loss_t)

    return model

def run_experiment(seed, num_epoch):
    model = create_model()

    dtype = torch.float32

    np.random.seed(seed)
    torch.manual_seed(seed)


    # toggle between learning rate and batch size values 
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
    model = train(model, optimizer, epochs=num_epoch)


    path = '/expanse/projects/nsg/external_users/public/arno/child_mind_abdu/'
    winLength = 2
    numChan = 24
    srate = 128
    feature = 'raw'
    one_channel = False
    
    # Testing
    test_data_balanced = load_data(path, 'test', winLength, numChan, srate, feature, False, 'v2')
    sample_acc1, subject_acc1 = test_model(model, test_data_balanced, path + 'test_subjIDs.csv', device, dtype)
    
    print(sample_acc1)
    print(subject_acc1)

    test_data_all_male = load_data(path, 'test', winLength, numChan, srate, feature,False, 'v3')
    sample_acc2, subject_acc2 = test_model(model, test_data_all_male, path + 'test_subjIDs_more_test.csv', device, dtype)

    print(sample_acc2)
    print(subject_acc2)
    
    return model

def test_model(model, test_data, subj_csv, device, dtype):
    # one-segment test
    loader_test = DataLoader(test_data, batch_size=70)
    per_sample_acc = check_accuracy(loader_test, model, device, dtype)

    # 40-segment test
    with open(subj_csv, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        subjIDs = [row[0] for row in spamreader]
    unique_subjs,indices = np.unique(subjIDs,return_index=True)

    iterable_test_data = list(iter(DataLoader(test_data, batch_size=1)))
    num_correct = []
    for subj,idx in zip(unique_subjs,indices):
    #     print(f'Subj {subj} - gender {iterable_test_data[idx][1]}')
        data = iterable_test_data[idx:idx+40]
        #print(np.sum([y for _,y in data]))
        assert 40 == np.sum([y for _,y in data]) or 0 == np.sum([y for _,y in data])
        preds = []
        correct = 0
        with torch.no_grad():
            for x,y in data:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                correct = y
                scores = model(x)
                _, pred = scores.max(1)
                preds.append(pred)
        final_pred = (torch.mean(torch.FloatTensor(preds)) > 0.5).sum()
        num_correct.append((final_pred == correct).sum())
    #print(len(num_correct))
    acc = float(np.sum(num_correct)) / len(unique_subjs)
    return per_sample_acc, acc


def test_all_seeds(model_path, model_type, feature, test_data, subjIDs_file, epoch, num_seed, device, dtype, logger):
    sample_acc = []
    subject_acc = []
    for s in range(num_seed):
        model = create_model(model_type, feature)
        model.load_state_dict(torch.load(f'{model_path}-seed{s}-epoch{epoch}'))
        model.to(device=device)
        sam_acc, sub_acc = test_model(model, test_data,subjIDs_file, device, dtype, logger)
        sample_acc.append(sam_acc)
        subject_acc.append(sub_acc)
        
    sample_acc = np.multiply(sample_acc,100)
    subject_acc = np.multiply(subject_acc,100)
    return sample_acc, subject_acc

def get_stats(arr):
    return np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
