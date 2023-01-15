import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from os import listdir
import numpy as np

filename = []
for mat_file in listdir('/expanse/projects/nsg/external_users/public/arno/mat_files'):
  filename.append(mat_file)

class Sound2MEGDataset(Dataset):
  def __init__(self, meg_files_path):
    #self.wav_files = wav_files
    self.meg_files_path = meg_files_path
  def __len__(self):
    return len(filename)*147
  def __getitem__(self, idx):
    idx_file, idx_sound = divmod(idx, 147)
    mat_file = loadmat(self.meg_files_path + filename[idx_file+1])
    mat_file = mat_file['data']
    MEG_Signal = np.float32(mat_file[:, :, idx_sound])
    MEG_Signal = np.reshape(MEG_Signal, (-1, 273, 3600))
    return torch.from_numpy(MEG_Signal)

Dataset = Sound2MEGDataset('/expanse/projects/nsg/external_users/public/arno/mat_files/')
Sample = Dataset[200]
print(Sample.shape)
