import torch 
import torch.nn as nn
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ArrayModel import *
from utils import *
from models import *
from tqdm import tqdm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train ULA with different SNR values')
parser.add_argument('--snr', type=float, required=True, help='Signal-to-noise ratio')
args = parser.parse_args()

m = 8 
d = 3
t = 200
n = 100000
snr = args.snr
distance = 0.1
lamda = 0.2
nbTrain = 5

array = ULA(m, lamda)
array.build_sensor_positions(distance)
array.build_array_manifold()

path = 'saved_models/'

loss_func = RMSPE(d, device=dev)

observations_test, angles_test = generate_data(n, t, d, snr, snr, array, True)
test_set = DATASET(observations_test, angles_test)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

MIN_DAMUSIC = 1000

for i in range(nbTrain):
    
    da_music = DA_MUSIC(m, d, array, dev)    
    da_music.load_state_dict(torch.load(path+'da_music_'+str(snr)+'dB_'+str(i)+'.pth', weights_only=True))
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            X, theta_true = data[0].to(dev), data[1].to(dev)
            theta_pred = da_music(X)
            loss = loss_func(theta_pred, theta_true)
            running_loss += loss.item()
    lossavg = running_loss/len(test_loader)
    if lossavg < MIN_DAMUSIC:
        MIN_DAMUSIC = lossavg
        torch.save(da_music.state_dict(), path+'da_music_'+str(snr)+'dB.pth')


MIN_DAMUSICv2 = 1000

for i in range(nbTrain):
    
    da_music_v2 = DA_MUSIC_v2(m, d, array, dev)    
    da_music_v2.load_state_dict(torch.load(path+'da_music_v2_'+str(snr)+'dB_'+str(i)+'.pth', weights_only=True))
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            X, theta_true = data[0].to(dev), data[1].to(dev)
            theta_pred = da_music_v2(X)
            loss = loss_func(theta_pred, theta_true)
            running_loss += loss.item()
    lossavg = running_loss/len(test_loader)
    if lossavg < MIN_DAMUSICv2:
        MIN_DAMUSICv2 = lossavg
        torch.save(da_music_v2.state_dict(), path+'da_music_v2_'+str(snr)+'dB.pth')


MIN_RNN = 1000

for i in range(nbTrain):
    
    rnn = RNN(m, d, dev)    
    rnn.load_state_dict(torch.load(path+'rnn_'+str(snr)+'dB_'+str(i)+'.pth', weights_only=True))
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            X, theta_true = data[0].to(dev), data[1].to(dev)
            theta_pred = rnn(X)
            loss = loss_func(theta_pred, theta_true)
            running_loss += loss.item()
    lossavg = running_loss/len(test_loader)
    if lossavg < MIN_RNN:
        MIN_RNN = lossavg
        torch.save(rnn.state_dict(), path+'rnn_'+str(snr)+'dB.pth')