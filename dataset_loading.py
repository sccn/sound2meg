# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from os import listdir
import os
import glob
import numpy as np
from sklearn.preprocessing import RobustScaler
import random
import csv

class Sound2MEGDataset(Dataset):
  def __init__(self, path):
    #self.wav_files = wav_files
    self.path = path
    self.sizes = np.array(loadmat(self.path + 'file_sizes.mat')['file_sizes'][0])
    self.filename = []
    self.subjects = []
    self.sizes = np.empty(0, dtype=int)
    self.subjects = []
    count = 0
    for subject in range(0, 126):
      if os.path.exists(self.path + 'MEG_Signals/S%03dT000.npy'%subject):
        files = glob.glob(self.path + 'MEG_Signals/S%03dT*.npy'%subject)
        self.sizes = np.append(self.sizes, len(files))
        self.subjects.append( '%03d'%subject )
        count = count + 1
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
    MEG_Signal = np.load(self.path + 'MEG_Signals/S'+self.subjects[idx_file]+'T%03d.npy'%idx_sound)
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
