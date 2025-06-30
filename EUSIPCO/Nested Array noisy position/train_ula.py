import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ArrayModel import *
from utils import *
from models import *
from tqdm import tqdm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = 8 
d = 3
t = 200
snr = 0
distance = 0.1
lamda = 0.2
nbTrain = 10

array = NestedArray1D(m, lamda)
array.build_sensor_positions(distance, [4, 4])

torch.manual_seed(30)
del_xy = torch.rand(2, 8) * 0.06 - 0.03
array.pertube(del_xy[0], del_xy[1])

array_nominal = NestedArray1D(m, lamda)
array_nominal.build_sensor_positions(distance, [4, 4])
array_nominal.build_array_manifold()

torch.manual_seed(torch.seed())

path = 'saved_models/'

n = 20000
lr = 1e-2
wd = 1e-9
batchSize = 256
nbEpoches = 300

train_func = RMSPE(d, device=dev)
valid_func = RMSPE(d, device=dev)


for i in range(nbTrain):

    observations, angles = generate_data(n, t, d, snr, snr, array, False, False)
    x_train, x_valid, theta_train, theta_valid = train_test_split(observations, angles, test_size=0.2)
    train_set = DATASET(x_train, theta_train)
    valid_set = DATASET(x_valid, theta_valid)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batchSize, shuffle=False)
    
    da_music = DA_MUSIC(m, d, array_nominal, dev)
    
    train(da_music, nbEpoches, lr, wd, 
          train_loader, valid_loader, 
          path+'da_music_'+str(snr)+'dB_nested_'+str(i)+'.pth', 
          train_func, valid_func)


for i in range(nbTrain):

    observations, angles = generate_data(n, t, d, snr, snr, array, False, False)
    x_train, x_valid, theta_train, theta_valid = train_test_split(observations, angles, test_size=0.2)
    train_set = DATASET(x_train, theta_train)
    valid_set = DATASET(x_valid, theta_valid)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batchSize, shuffle=False)
    
    da_music_v2 = DA_MUSIC_v2(m, d, array_nominal, dev)
    
    train(da_music_v2, nbEpoches, lr, wd, 
          train_loader, valid_loader, 
          path+'da_music_v2_'+str(snr)+'dB_nested_'+str(i)+'.pth', 
          train_func, valid_func)
    

for i in range(nbTrain):

    observations, angles = generate_data(n, t, d, snr, snr, array, False, False)
    x_train, x_valid, theta_train, theta_valid = train_test_split(observations, angles, test_size=0.2)
    train_set = DATASET(x_train, theta_train)
    valid_set = DATASET(x_valid, theta_valid)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batchSize, shuffle=False)

    rnn = RNN(m, d, dev)

    train(rnn, nbEpoches, lr, wd, train_loader, valid_loader, 
          path+'rnn_'+str(snr)+'dB_nested_'+str(i)+'.pth', 
          train_func, valid_func)