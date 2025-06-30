import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import *
from models import *
from tqdm import tqdm

import matplotlib.pyplot as plt

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = 3
m = 8
t = 200
n = 100000
lamda = 0.2
distance = 0.1

array = ULA(m, lamda)
array.build_sensor_positions(distance)

seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
SNRs = [-10, -5, 0, 5, 10, 15, 20]
colors = ['blue', 'orange', 'green', 'red']
names = ['MUSIC', 'DA-MUSIC', 'Proposed Modified DA-MUSIC', 'Proposed DD Model']
rmspe = {'MUSIC': torch.zeros(len(seeds), len(SNRs)), 
         'DA-MUSIC': torch.zeros(len(seeds), len(SNRs)), 
         'Proposed Modified DA-MUSIC': torch.zeros(len(seeds), len(SNRs)), 
         'Proposed DD Model': torch.zeros(len(seeds), len(SNRs))}

for j, seed in enumerate(seeds):

    for k, snr in enumerate(SNRs):

        torch.manual_seed(seed)
        del_x = torch.rand(m) * 0.04 - 0.02
        del_y = torch.zeros(m)
        array.pertube(del_x, del_y)
        torch.seed()

        array_nominal = ULA(m, lamda)
        array_nominal.build_sensor_positions(distance)
        array_nominal.build_array_manifold()

        loss_cpu = RMSPE(d, 'cpu')
        loss_cuda = RMSPE(d, 'cuda')

        observations, angles = generate_data(n, t, d, snr, snr, array, False, False)

        music = MUSIC(d, array_nominal, -torch.pi/2, torch.pi/2, 360)
        results_music = []
        for i in range(observations.shape[0]):
            theta_est, _ = music.estimate(observations[i].T)
            results_music.append(theta_est)
        results_music = torch.stack(results_music, dim=0)
        rmspe['MUSIC'][j, k] = loss_cpu(results_music, angles)


        test_set = DATASET(observations, angles)
        test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

        da_music = DA_MUSIC(m, d, array_nominal, dev)
        rmspe['DA-MUSIC'][j, k] = test(da_music, test_loader, 'saved_models/da_music_'+str(snr)+'dB_seed_'+str(seed)+'.pth', loss_cuda, dev)

        da_music_v2 = DA_MUSIC_v2(m, d, array_nominal, dev)
        rmspe['Proposed Modified DA-MUSIC'][j, k] = test(da_music_v2, test_loader, 'saved_models/da_music_v2_'+str(snr)+'dB_seed_'+str(seed)+'.pth', loss_cuda, dev)

        rnn = RNN(m, d, dev)
        rmspe['Proposed DD Model'][j, k] = test(rnn, test_loader, 'saved_models/rnn_'+str(snr)+'dB_seed_'+str(seed)+'.pth', loss_cuda, dev)


rmspe['MUSIC'] = torch.mean(rmspe['MUSIC'], dim=0)
rmspe['DA-MUSIC'] = torch.mean(rmspe['DA-MUSIC'], dim=0)
rmspe['Proposed Modified DA-MUSIC'] = torch.mean(rmspe['Proposed Modified DA-MUSIC'], dim=0)
rmspe['Proposed DD Model'] = torch.mean(rmspe['Proposed DD Model'], dim=0)

plt.rcParams['figure.dpi'] = 200

for method, color in zip(names, colors):
    plt.plot(SNRs, rmspe[method]/torch.pi*180, marker='o', label=method, color=color)

plt.yscale("log")
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("RMSPE (degree)", fontsize=12)
plt.xticks(SNRs)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('m8d3_perturbation_log_degree.png')