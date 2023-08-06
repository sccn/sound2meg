from utils import *
import torch

path = '/expanse/projects/nsg/external_users/public/arno/child_mind_abdu/'
winLength = 2
numChan = 24
srate = 128
feature = 'raw'
one_channel = False

role = 'train'
train_data = load_data(path, role, winLength, numChan, srate, feature, one_channel)
print(f'X_train shape: {len(train_data)}, {train_data[0][0].shape}')
print(f'Y_train shape: {len(train_data)}, {train_data[0][1].shape}')

role = 'val'
val_data = load_data(path, role, winLength, numChan, srate, feature, one_channel)
print(f'X_val shape: {len(val_data)}, {val_data[0][0].shape}')
print(f'Y_val shape: {len(val_data)}, {val_data[0][1].shape}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 70 
loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=batch_size)

for s in range(1):
    model = run_experiment(s, 70)
