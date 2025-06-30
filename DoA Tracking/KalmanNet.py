import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import *
from Kalman import *

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class KalmanNet(nn.Module):

    def __init__(self, 
                 array: ArrayModel, 
                 d: int,
                 Ts: float,
                 Ns: int,
                 device: str):
        
        super(KalmanNet, self).__init__()

        self.array = array
        self.m = self.array.m
        self.d = d
        self.Ts = Ts
        self.Ns = Ns
        self.F = torch.Tensor([[[1, Ts],
                                [0, 1]]]).repeat(d, 1, 1).to(device)
        self.H = torch.Tensor([[[1, 0]]]).repeat(d, 1, 1).to(device)
        self.KG = KalmanGain(d=d, m=2, n=1, device=device)
        
        self.device = device
        self.array.array = self.array.array.to(device)

    def initialize(self, X: torch.Tensor):
        
        cov_0 = compute_covariance(X[:, :self.Ns, :])
        theta_0, signal_cov, signal_sigma2 = DoA_Estimation_batch(Rx=cov_0, d=self.d, array=self.array, solver=ESPRIT)
        cov_1 = compute_covariance(X[:, self.Ns:, :])
        theta_1, _, _                      = DoA_Estimation_batch(Rx=cov_1, d=self.d, array=self.array, solver=ESPRIT)

        self.signal_cov = signal_cov.to(self.device)
        self.signal_sigma2 = signal_sigma2.to(self.device)

        self.x = torch.stack((theta_1, (theta_1 - theta_0) / self.Ts), dim=2).unsqueeze(3).to(self.device)
        self.x_prev = self.x    
        self.x_prev_pred = self.x

        self.R_x_prev = cov_1

        self.KG.initialize(Q=torch.zeros(X.shape[0], self.d, 4),
                           Sigma=torch.zeros(X.shape[0], self.d, 4),
                           S=torch.zeros(X.shape[0], self.d, 1))

        return theta_0, theta_1
    
    def estimate(self, X: torch.Tensor):

        x_pred = self.F @ self.x
        A_pred = self.array.get_steering_vector(theta=(self.H @ x_pred).squeeze(-1, -2))
        R_pred = A_pred @ self.signal_cov @ A_pred.transpose(1, 2).conj() \
               + torch.diag_embed(self.signal_sigma2.unsqueeze(1).repeat(1, self.m))
        R_x = compute_covariance(X)

        del_y_til = compute_angle_displacement(delta_cov=R_x-self.R_x_prev, theta=(self.H @ self.x).squeeze(-1, -2), 
                                               S=self.signal_cov.detach(), array=self.array).unsqueeze(-1)
        
        del_y = compute_angle_displacement(delta_cov=R_x-R_pred, theta=(self.H @ x_pred).squeeze(-1, -2), 
                                           S=self.signal_cov.detach(), array=self.array).unsqueeze(-1)
        
        # del_y_til = compute_angle_displacement_ver3(new_cov=R_x, theta=(self.H @ self.x).squeeze(-1, -2), array=self.array).unsqueeze(-1)
        
        # del_y = compute_angle_displacement_ver3(new_cov=R_x, theta=(self.H @ x_pred).squeeze(-1, -2), array=self.array).unsqueeze(-1)

        del_x_til = (self.x - self.x_prev).squeeze(-1)

        del_x_hat = (self.x - self.x_prev_pred).squeeze(-1)
        
        K = self.KG(del_y_til, del_y, del_x_til, del_x_hat)
        K = K.reshape(-1, self.d, 2, 1)

        self.x_prev_pred = x_pred
        self.x_prev = self.x
        self.x = x_pred + (K @ del_y.unsqueeze(-1))
        
        self.R_x_prev = R_x
        

        return (self.H @ self.x).squeeze(-1, -2)
    
    def forward(self, X: torch.Tensor):

        self.initialize(X[:, :2 * self.Ns, :])
        y = []
        for i in range(2, int(X.shape[1]/self.Ns)):
            y.append(self.estimate(X[:, self.Ns * i : self.Ns * (i + 1)]))
        y = torch.stack(y, dim=1)

        return y
    


class KalmanGain(nn.Module):

    def __init__(self, 
                 d: int, m: int, n: int, 
                 in_mult: int = 20, out_mult: int = 20, 
                 device: str = dev):
        
        super(KalmanGain, self).__init__()

        self.d = d
        self.m = m
        self.n = n
        self.device = device

        self.GRU1 = nn.ModuleList([nn.GRU(input_size=in_mult*m, hidden_size=m*m, batch_first=True, device=device) for _ in range(d)])
        
        self.GRU2 = nn.ModuleList([nn.GRU(input_size=in_mult*m+m*m, hidden_size=m*m, batch_first=True, device=device) for _ in range(d)])
        
        self.GRU3 = nn.ModuleList([nn.GRU(input_size=in_mult*2*n+n*n, hidden_size=n*n, batch_first=True, device=device) for _ in range(d)])
        
        self.fc1 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=m, out_features=in_mult*m, device=device), 
                                                nn.ReLU()) for _ in range(d)])
        self.fc2 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=m, out_features=in_mult*m, device=device), 
                                                nn.ReLU()) for _ in range(d)])
        self.fc3 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=m*m, out_features=n*n, device=device), 
                                                nn.ReLU()) for _ in range(d)])
        self.fc4 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=2*n, out_features=in_mult*2*n, device=device), 
                                                nn.ReLU()) for _ in range(d)])
        self.fc5 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=n*n+m*m, out_features=out_mult*(n*n+m*m), device=device),
                                                nn.ReLU(), 
                                                nn.Linear(in_features=out_mult*(n*n+m*m), out_features=m*n, device=device)) for _ in range(d)])
        self.fc6 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=n*n+m*n, out_features=m*m, device=device), 
                                                nn.ReLU()) for _ in range(d)])
        self.fc7 = nn.ModuleList([nn.Sequential(nn.Linear(in_features=2*m*m, out_features=m*m, device=device), 
                                                nn.ReLU()) for _ in range(d)])
        
    def initialize(self, 
                   Q: torch.Tensor, 
                   Sigma: torch.Tensor, 
                   S: torch.Tensor):

        self.Q = Q.to(self.device)
        self.Sigma = Sigma.to(self.device)
        self.S = S.to(self.device)

    def forward(self, 
                del_y_til: torch.Tensor, 
                del_y: torch.Tensor, 
                del_x_til: torch.Tensor, 
                del_x_hat: torch.Tensor) -> torch.Tensor:
        
        list_Q = []
        list_Sigma = []
        list_S = []
        list_K = []

        for i in range(self.d):

            input_gru1 = self.fc1[i](del_x_hat[:, i]).unsqueeze(1)
            hidden_gru1 = self.Q[:, i, :].unsqueeze(0).contiguous()
            input_gru1 = F.normalize(input=input_gru1, p=2.0, dim=2, eps=1e-6)
            _, Qi = self.GRU1[i](input_gru1, hidden_gru1)
            Qi = Qi.squeeze(0)
            
            input_gru2 = torch.cat((Qi, self.fc2[i](del_x_til[:, i])), dim=1).unsqueeze(1)
            hidden_gru2 = self.Sigma[:, i, :].unsqueeze(0).contiguous()
            input_gru2 = F.normalize(input=input_gru2, p=2.0, dim=2, eps=1e-6)
            _, Sigmai = self.GRU2[i](input_gru2, hidden_gru2)
            Sigmai = Sigmai.squeeze(0)
            
            input_gru3 = torch.cat((self.fc3[i](Sigmai), self.fc4[i](torch.cat((del_y_til[:, i], del_y[:, i]), dim=1))), dim=1).unsqueeze(1)
            hidden_gru3 = self.S[:, i, :].unsqueeze(0).contiguous()
            input_gru3 = F.normalize(input=input_gru3, p=2.0, dim=2, eps=1e-6)
            _, Si = self.GRU3[i](input_gru3, hidden_gru3)
            Si = Si.squeeze(0)

            Ki = self.fc5[i](torch.cat((Sigmai, Si), dim=1))
            
            list_K.append(Ki)
            list_Q.append(Qi)
            list_S.append(Si)
            list_Sigma.append(self.fc7[i](torch.cat((Sigmai, self.fc6[i](torch.cat((Si, Ki), dim=1))), dim=1)))

        self.Q = torch.stack(list_Q, dim=1)
        self.Sigma = torch.stack(list_Sigma, dim=1)
        self.S = torch.stack(list_S, dim=1)
        
        K = torch.stack(list_K, dim=1)
        
        return K
    


# def train(net: KalmanNet, 
#           X_train: torch.Tensor, y_train: torch.Tensor, 
#         #   X_valid: torch.Tensor, y_valid: torch.Tensor,
#           batchSize: int, nbEpochs: int, lr: float, wd: float, 
#           saved_model: str):
    
#     train_loader = DataLoader(DATA(X_train, y_train, Ns=net.Ns), batch_size=batchSize, shuffle=True)
#     # valid_loader = DataLoader(DATA(X_valid, y_valid), batch_size=batchSize, shuffle=False)

#     opt = optim.Adam(params=net.parameters(), lr=lr, weight_decay=wd)

#     Loss = []

#     # Val = []
#     # bestVal = 10000

#     for i in tqdm(range(nbEpochs)):

#         running_loss = 0.0
#         for data in train_loader:
#             input, target = data[0].to(net.device), data[1].to(net.device)
#             opt.zero_grad()
#             pred = net(input)
#             loss = nn.MSELoss()(pred, target[:, 2:])
#             loss.backward()
#             opt.step()
#             running_loss += loss.item()
#         Loss.append(running_loss/len(train_loader))
#         print("Loss {} is {}".format(i, Loss[-1]))

#     torch.save(net.state_dict(), saved_model)

#         # with torch.no_grad():
#         #     running_loss = 0.0
#         #     for data in valid_loader:
#         #         input, target = data[0].to(dev), data[1].to(dev)
#         #         pred = net(input)
#         #         loss = nn.MSELoss()(pred, target[:, 2:])
#         #         running_loss += loss.item()
#         #     Val.append(running_loss/len(valid_loader))
#         #     if Val[i] < bestVal:
#         #         bestVal = Val[i]
#         #         torch.save(net.state_dict(), saved_model)

#     return Loss



# def test(net: KalmanNet, 
#          saved_model: str,
#          X_test: torch.Tensor, y_test: torch.Tensor):
    
#     net.load_state_dict(torch.load(saved_model))
#     pred = net(X_test)
#     loss = nn.MSELoss()(pred, y_test[:, 2:])
#     return loss