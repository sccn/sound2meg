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
  def __init__(self, path, embedding_type):
    #self.wav_files = wav_files
    self.path = path
#     self.sizes = np.array(loadmat(self.path + 'file_sizes.mat')['file_sizes'][0]) # dutruong - unused
#     self.filename = []  # dutruong - unused
    self.sizes = np.empty(0, dtype=int) # dutruong - for __len__ to work
    self.subjects = []
    self.embedding_type = embedding_type
    if embedding_type == 'mel':
      self.subpath = 'arno/Mel_Embedding120/mel_'
    elif embedding_type == 'Wav2Vec':
      self.subpath = 'jiyul/wavEmbedding/'
    for subject in range(0, 126):
      if os.path.exists(self.path + 'arno/MEG_Signals/S%03dT000.npy'%subject):
        files = glob.glob(self.path + 'arno/MEG_Signals/S%03dT*.npy'%subject) 
        self.sizes = np.append(self.sizes, len(files)) # dtruong - number of (brain) samples. So not the file size
        self.subjects.append( '%03d'%subject )
#         count = count + 1 # dtruong - unused
  def __len__(self):
    return sum(self.sizes)
  def __getitem__(self, idx):
    file_idx_upper_bounds = np.cumsum(self.sizes) # dutruong - inclusive lower bound
    idx_file = np.argwhere(file_idx_upper_bounds > idx)[0][0] # dutruong - first upper bound to contain idx (exclusive) is the file index. np.argwhere returns array (here of size 1), hence the 2nd 0 indexing
#Taking a cummulative sum of all the sizes
#This way we will know the ranges of the different MEG Signal samples in the context of the entire dataset. # dutruong - to determine which elem of self.sizes the idx corresponds to. This means samples are consecutive. Have to make sure shuffling.
#For example, say the sizes of the mat files are [3, 5, 2]
#Using the cummulative function, we get [3, 8, 10]
    if idx_file == 0:                             #Accordingly, we see which sample is in which range by finding the smallest value in the cumsum larger than idx                           
      idx_sound = idx
    else:
      idx_sound = idx - file_idx_upper_bounds[idx_file - 1]    
    MEG_Signal = np.load(self.path + 'arno/MEG_Signals/S'+self.subjects[idx_file]+'T%03d.npy'%idx_sound)
    with open(self.path + 'arno/MEG_Signals/audios.csv', 'r') as f:
      audios = list(csv.reader(f))
      audio_file = audios[idx_file][idx_sound][2:5] # file name "['186.wav']"
    #Finding the associated audiofile, and extracting only the number of the file out of it
    #audio_file = mat_file['audiofiles'][0, idx_sound][0][0:3]
    #The filenames are in the format 'mel_<number>.npy' so importing accordingly
    Sound_Signal = np.load(self.path + self.subpath + audio_file + '.npy')
    if self.embedding_type == 'Wav2Vec':
      Sound_Signal = Sound_Signal[0]
    #mat_file = mat_file['data']
    MEG_Signal = np.float32(MEG_Signal[:273, :]) # dutruong - some dataset have more channels
    #Robust Scaler
    scaler = RobustScaler().fit(MEG_Signal)
    MEG_Signal = scaler.transform(MEG_Signal)
    MEG_Signal = np.minimum(MEG_Signal, 20)
    return torch.from_numpy(MEG_Signal), torch.from_numpy(Sound_Signal) , idx_file
