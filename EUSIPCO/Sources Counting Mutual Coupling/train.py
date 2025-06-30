import torch , argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from models import *
from tqdm import tqdm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train ULA with different SNR values')
parser.add_argument('--snr', type=float, required=True, help='Signal-to-noise ratio')
args = parser.parse_args()

dmin = 2
dmax = 5
m = 8
t = 200
snr = args.snr
lamda = 0.2
radius = 0.1
mc_range = 3

n = 50000
lr = 1e-3
wd = 1e-9
batchSize = 32
nbEpoches = 250

array = UCA(m=m, lamda=lamda)
array.build_sensor_positions(radius=radius)
array.build_array_manifold()

torch.manual_seed(30)
mc_coef = torch.zeros(5, dtype=torch.complex64)
mc_coef[:mc_range] = torch.rand(mc_range, dtype=torch.complex64)
C = build_symmetric_circulant_toeplitz(mc_coef) 
C = C - torch.diag(torch.diag(C)) + torch.eye(m, dtype=torch.complex64)
torch.manual_seed(torch.seed())

observations, angles, labels = generate_data(n, t, dmin, dmax, snr, snr, array, False, C)

x_train, x_valid, theta_train, theta_valid, label_train, label_valid = train_test_split(observations, angles, labels, test_size=0.3)

train_set = DATASET(x_train, theta_train, label_train)
valid_set = DATASET(x_valid, theta_valid, label_valid)

train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batchSize, shuffle=False)


# #################### DA-MUSIC ####################


# model = DA_MUSIC(dmin, dmax, array).to(dev)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# loss_func1 = RMSPE_varied_nbSources(dmin, dmax, dev)
# loss_func2 = nn.CrossEntropyLoss()
# Loss, Loss_d, Loss_theta, Val_d, Val_theta = [], [], [], [], []
# bestVal_d = 0.0
# bestVal_theta = 1000.0


# for i in tqdm(range(nbEpoches)):

#     running_loss = 0.0
#     running_loss_d = 0.0
#     running_loss_theta = 0.0

#     for data in train_loader:   
#         X, theta_true, label_true = data[0].to(dev), data[1].to(dev), data[2].to(dev)
#         optimizer.zero_grad()
#         theta_pred, label_pred = model(X)
#         loss_theta = loss_func1.calculate(theta_pred, theta_true) 
#         loss_theta.backward()
#         loss_d = loss_func2(label_pred, label_true)
#         loss_d.backward()
#         optimizer.step()
#         running_loss_theta += loss_theta.item()
#         running_loss_d += loss_d.item()
#         running_loss += loss_theta.item() + loss_d.item()
    
#     Loss.append(running_loss/len(train_loader))
#     Loss_d.append(running_loss_d/len(train_loader))
#     Loss_theta.append(running_loss_theta/len(train_loader))

#     with torch.no_grad():

#         running_acc_d = 0.0
#         running_loss_theta = 0.0

#         for data in valid_loader:
#             X, theta_true, label_true = data[0].to(dev), data[1].to(dev), data[2].to(dev)
#             theta_pred, label_pred = model(X)
#             loss_theta = loss_func1.calculate(theta_pred, theta_true) 
#             acc_d = (torch.argmax(label_pred, dim=1) == torch.argmax(label_true, dim=1)).float().mean()
#             running_loss_theta += loss_theta.item()
#             running_acc_d += acc_d.item()
        
#         Val_d.append(running_acc_d/len(valid_loader))
#         Val_theta.append(running_loss_theta/len(valid_loader))

#         # if Val_d[-1] > bestVal_d:
            
#         #     bestVal_d = Val_d[-1]
#         #     torch.save(model.state_dict(), "saved_model.pth")

#         if Val_theta[-1] < bestVal_theta:
#             bestVal_theta = Val_theta[-1]
#             torch.save(model.state_dict(), "saved_models/damusic_{}dB.pth".format(snr))
#             count = 0
#         else:
#             count += 1

#         if count == 20:
#             model.load_state_dict(torch.load("saved_models/damusic_{}dB.pth".format(snr), weights_only=True))


#         print("Iteration {}: RMSPE = {}, Accuracy = {}".format(i, Val_theta[-1], Val_d[-1]))

# print("Finish training DA-MUSIC")


# #################### DA-MUSIC v2 ####################


# model = DA_MUSIC_v2(dmin, dmax, array).to(dev)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# loss_func1 = RMSPE_varied_nbSources(dmin, dmax, dev)
# loss_func2 = nn.CrossEntropyLoss()
# Loss, Loss_d, Loss_theta, Val_d, Val_theta = [], [], [], [], []
# bestVal_d = 0.0
# bestVal_theta = 1000.0


# for i in tqdm(range(nbEpoches)):

#     running_loss = 0.0
#     running_loss_d = 0.0
#     running_loss_theta = 0.0

#     for data in train_loader:

#         X, theta_true, label_true = data[0].to(dev), data[1].to(dev), data[2].to(dev)
#         optimizer.zero_grad()
#         theta_pred, label_pred = model(X)
#         loss_theta = loss_func1.calculate(theta_pred, theta_true) 
#         loss_theta.backward()
#         loss_d = loss_func2(label_pred, label_true)
#         loss_d.backward()
#         optimizer.step()
#         running_loss_theta += loss_theta.item()
#         running_loss_d += loss_d.item()
#         running_loss += loss_theta.item() + loss_d.item()
    
#     Loss.append(running_loss/len(train_loader))
#     Loss_d.append(running_loss_d/len(train_loader))
#     Loss_theta.append(running_loss_theta/len(train_loader))

#     with torch.no_grad():

#         running_acc_d = 0.0
#         running_loss_theta = 0.0

#         for data in valid_loader:
            
#             X, theta_true, label_true = data[0].to(dev), data[1].to(dev), data[2].to(dev)
#             theta_pred, label_pred = model(X)
#             loss_theta = loss_func1.calculate(theta_pred, theta_true) 
#             acc_d = (torch.argmax(label_pred, dim=1) == torch.argmax(label_true, dim=1)).float().mean()
#             running_loss_theta += loss_theta.item()
#             running_acc_d += acc_d.item()
        
#         Val_d.append(running_acc_d/len(valid_loader))
#         Val_theta.append(running_loss_theta/len(valid_loader))

#         # if Val_d[-1] > bestVal_d:
            
#         #     bestVal_d = Val_d[-1]
#         #     torch.save(model.state_dict(), "saved_model.pth")

#         if Val_theta[-1] < bestVal_theta:
#             bestVal_theta = Val_theta[-1]
#             torch.save(model.state_dict(), "saved_models/damusic_v2_{}dB.pth".format(snr))
#             count = 0
#         else:
#             count += 1

#         if count == 20:
#             model.load_state_dict(torch.load("saved_models/damusic_v2_{}dB.pth".format(snr), weights_only=True))


#         print("RMSPE = {}, Accuracy = {}".format(Val_theta[-1], Val_d[-1]))


# print("Finish training DA-MUSIC v2")


# #################### RNN ####################


# model = RNN(m, dmin, dmax).to(dev)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# loss_func1 = RMSPE_varied_nbSources(dmin, dmax, dev)
# loss_func2 = nn.CrossEntropyLoss()
# Loss, Loss_d, Loss_theta, Val_d, Val_theta = [], [], [], [], []
# bestVal_d = 0.0
# bestVal_theta = 1000.0


# for i in tqdm(range(nbEpoches)):

#     running_loss = 0.0
#     running_loss_d = 0.0
#     running_loss_theta = 0.0

#     for data in train_loader:

#         X, theta_true, label_true = data[0].to(dev), data[1].to(dev), data[2].to(dev)
#         optimizer.zero_grad()
#         theta_pred, label_pred = model(X)
#         loss_theta = loss_func1.calculate(theta_pred, theta_true) 
#         loss_theta.backward()
#         loss_d = loss_func2(label_pred, label_true)
#         loss_d.backward()
#         optimizer.step()
#         running_loss_theta += loss_theta.item()
#         running_loss_d += loss_d.item()
#         running_loss += loss_theta.item() + loss_d.item()
    
#     Loss.append(running_loss/len(train_loader))
#     Loss_d.append(running_loss_d/len(train_loader))
#     Loss_theta.append(running_loss_theta/len(train_loader))

#     with torch.no_grad():

#         running_acc_d = 0.0
#         running_loss_theta = 0.0

#         for data in valid_loader:
            
#             X, theta_true, label_true = data[0].to(dev), data[1].to(dev), data[2].to(dev)
#             theta_pred, label_pred = model(X)
#             loss_theta = loss_func1.calculate(theta_pred, theta_true) 
#             acc_d = (torch.argmax(label_pred, dim=1) == torch.argmax(label_true, dim=1)).float().mean()
#             running_loss_theta += loss_theta.item()
#             running_acc_d += acc_d.item()
        
#         Val_d.append(running_acc_d/len(valid_loader))
#         Val_theta.append(running_loss_theta/len(valid_loader))

#         # if Val_d[-1] > bestVal_d:
            
#         #     bestVal_d = Val_d[-1]
#         #     torch.save(model.state_dict(), "saved_model.pth")

#         if Val_theta[-1] < bestVal_theta:
#             bestVal_theta = Val_theta[-1]
#             torch.save(model.state_dict(), "saved_models/rnn_{}dB.pth".format(snr))
#             count = 0
#         else:
#             count += 1

#         if count == 20:
#             model.load_state_dict(torch.load("saved_models/rnn_{}dB.pth".format(snr), weights_only=True))


#         print("RMSPE = {}, Accuracy = {}".format(Val_theta[-1], Val_d[-1]))


# print("Finish training RNN")

#################### ECNet ####################


model = ECNet(m=m, dmin=dmin, dmax=dmax, nbLayers=10, nbNeurons=10, device=dev)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
loss_func = nn.CrossEntropyLoss()
Loss, Val = [], []
bestVal = 0.0

for i in tqdm(range(nbEpoches)):

    running_loss = 0.0

    for data in train_loader:
        X, label_true = data[0].to(dev), data[2].to(dev)
        optimizer.zero_grad()
        label_pred = model(X)
        loss = loss_func(label_pred, label_true)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    Loss.append(running_loss/len(train_loader))

    with torch.no_grad():

        running_acc = 0.0

        for data in valid_loader:
            X, label_true = data[0].to(dev), data[2].to(dev)
            label_pred = model(X)
            acc = (torch.argmax(label_pred, dim=1) == torch.argmax(label_true, dim=1)).float().mean()
            running_acc += acc.item()
        
        Val.append(running_acc/len(valid_loader))

        if Val[-1] > bestVal:
            bestVal = Val[-1]
            torch.save(model.state_dict(), "saved_models/ecnet_{}dB.pth".format(snr))
            count = 0
        else:
            count += 1

        if count == 20:
            model.load_state_dict(torch.load("saved_models/ecnet_{}dB.pth".format(snr), weights_only=True))

    print("Iteration {}: Accuracy = {}".format(i, Val[-1]))

print("Finish training ECNet")