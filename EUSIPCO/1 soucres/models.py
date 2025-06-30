import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from ArrayModel import *

from tqdm import tqdm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DA_MUSIC(nn.Module): 
    
    def __init__(self, m: int, d: int, array, device=dev):
        
        super().__init__()

        self.m = m
        self.d = d
        self.array_manifold = array.array_manifold.to(device)

        self.bn = nn.BatchNorm1d(2*self.m, device=device)
        self.rnn = nn.GRU(input_size=2*self.m, hidden_size=2*self.m, num_layers=1, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=2*self.m, out_features=2*self.m*self.m, device=device)
        self.mlp = nn.Sequential(nn.Linear(in_features=array.nbSamples_spectrum, out_features=2*self.m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*self.m, out_features=self.d, device=device))

    def forward(self, x):
        
        x = torch.cat((torch.real(x), torch.imag(x)), dim=-1)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        _, x = self.rnn(x)
        x = self.fc(x[-1])
        x = x.reshape(-1, 2, self.m, self.m)
        x = x[:, 0, :, :] + 1j * x[:, 1, :, :]
        vals, vecs = torch.linalg.eig(x)
        idx = torch.sort(torch.abs(vals), dim=1)[1].unsqueeze(dim=1).repeat(repeats=(1, self.m, 1))
        vecs = torch.gather(vecs, dim=2, index=idx)
        E = vecs[:, :, :(self.m - self.d)]
        spectrum = 1 / torch.norm(E.conj().transpose(-2, -1) @ self.array_manifold, dim=-2) ** 2
        y = self.mlp(spectrum)
        
        return y



class DA_MUSIC_v2(nn.Module): 
    
    def __init__(self, m: int, d: int, array, device=dev):
        
        super().__init__()

        self.m = m
        self.d = d
        self.array_manifold = array.array_manifold.to(device)
        self.device = device

        self.bn = nn.BatchNorm1d(2*self.m, device=device)
        self.rnn = nn.GRU(input_size=2*self.m, hidden_size=2*self.m, num_layers=1, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=2*self.m, out_features=2*self.m*self.m, device=device)
        self.mlp = nn.Sequential(nn.Linear(in_features=array.nbSamples_spectrum, out_features=2*self.m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*self.m, out_features=self.d, device=device))

    def forward(self, x):
        
        x_ = torch.cat((torch.real(x), torch.imag(x)), dim=-1)
        x_ = self.bn(x_.transpose(1, 2)).transpose(1, 2)
        _, x_ = self.rnn(x_)
        x_ = self.fc(x_[-1])
        x_ = x_.reshape(-1, 2, self.m, self.m)
        x_ = torch.complex(x_[:, 0, :, :], x_[:, 1, :, :])
        Rx = x_ @ x_.conj().transpose(1, 2)
        vals, vecs = torch.linalg.eigh(Rx)
        idx = torch.sort(torch.abs(vals), dim=1)[1].unsqueeze(dim=1).repeat(repeats=(1, self.m, 1))
        vecs = torch.gather(vecs, dim=2, index=idx)
        E = vecs[:, :, :(self.m - self.d)]
        spectrum = 1 / torch.norm(E.conj().transpose(-2, -1) @ self.array_manifold, dim=-2) ** 2
        y = self.mlp(spectrum)
        
        return y
    

class minGRU(nn.Module):
    def __init__(self,input_size:int,hidden_size:int):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 

        self.linear_h = nn.Linear(input_size,hidden_size)
        self.linear_z = nn.Linear(input_size,hidden_size)
    
    @staticmethod
    def g(x:torch.Tensor) -> torch.Tensor:
        """
        The continuous function g ensures that ht ← g(Lineardh (xt)) is positive
        """
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
    
    @staticmethod
    def log_g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))

    @staticmethod
    def parallel_scan_log(log_coeffs:torch.Tensor, log_values:torch.Tensor) -> torch.Tensor:
        """
        log_coeffs: [batch_size, seq_len, input_size]
        log_values: [batch_size, seq_len + 1, input_size]

        """
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]

    def sequential_forward(self, x_t:torch.Tensor, h_prev:torch.Tensor=None) -> torch.Tensor:
        """
        x_t     : [batch_size, input_size]
        h_prev  : [batch_size, hidden_size]
        """
        if(h_prev is None):
            h_prev = self.g(torch.zeros((x_t.size(0),self.hidden_size),device=x_t.device))

        z = torch.sigmoid(self.linear_z(x_t)) 
        h_tilde = self.g(self.linear_h(x_t)) 
        h_t = (1 - z) * h_prev + z * h_tilde

        return h_t

    def forward(self, x:torch.Tensor, h_0:torch.Tensor=None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_size]
        h_0: [batch_size, 1, hidden_size]
        """
        if(h_0 is None):
            h_0 = torch.zeros((x.size(0),1,self.hidden_size),device=x.device)


        k = self.linear_z(x) 
        log_coeffs = -F.softplus(k)
        log_z = -F.softplus(-k)
        log_tilde_h = self.log_g(self.linear_h(x)) 

        log_h_0 = self.log_g(h_0)
        h = self.parallel_scan_log(log_coeffs,torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        return h #(batch_size,seq_size,hidden_size)
    

class RNN2(nn.Module): 
    
    def __init__(self, m: int, d: int, device=dev):
        
        super().__init__()

        self.m = m
        self.d = d
        self.device = device

        self.bn = nn.BatchNorm1d(2*m, device=device)
        self.rnn = nn.GRU(input_size=2*m, hidden_size=2*m, num_layers=1, batch_first=True, device=device)

        self.mlp = nn.Sequential(nn.Linear(in_features=2*m, out_features=2*m*m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*m*m, out_features=2*m*m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*m*m, out_features=self.d, device=device))

    def forward(self, x):
        
        x = torch.cat((torch.real(x), torch.imag(x)), dim=-1)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        _, x = self.rnn(x)
        y = self.mlp(x[-1])

        return y


class RNN(nn.Module): 
    
    def __init__(self, m: int, d: int, device=dev):
        
        super().__init__()

        self.m = m
        self.d = d
        self.device = device

        self.bn = nn.BatchNorm1d(2*m, device=device)
        self.rnn = nn.GRU(input_size=2*m, hidden_size=2*m, num_layers=1, batch_first=True, device=device)

        self.mlp = nn.Sequential(nn.Linear(in_features=2*m, out_features=2*m*m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*m*m, out_features=2*m*m, device=device), nn.ReLU(),
                                 nn.Linear(in_features=2*m*m, out_features=self.d, device=device))

    def forward(self, x):
        
        x = torch.cat((torch.real(x), torch.imag(x)), dim=-1)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        _, x = self.rnn(x)
        y = self.mlp(x[-1])

        return y
    


def train(model, nbEpoches, lr, wd, train_loader, valid_loader, model_name, device, loss_func, valid_func):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    Loss, Val = [], []
    bestVal = 1000

    for i in tqdm(range(nbEpoches)):
        # Train
        model.train()
        running_loss = 0.0
        for data in train_loader:
            X, theta_true = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            theta_pred = model(X) 
            loss = loss_func(theta_pred, theta_true) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        Loss.append(running_loss/len(train_loader))

        scheduler.step()

        # Validation 
        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            for data in valid_loader:
                X, theta_true = data[0].to(device), data[1].to(device)
                theta_pred = model(X)
                loss = valid_func(theta_pred, theta_true)
                running_loss += loss.item()
            
            Val.append(running_loss/len(valid_loader))

            if Val[i] < bestVal:
                bestVal = Val[i]
                torch.save(model.state_dict(), model_name)

        print("Iteration {}: loss training = {}, best validation = {}".format(i, Loss[-1], bestVal))

    return Loss, Val



def test(model, test_loader, model_name, test_func, device):
    
    if model_name is not None:
        model.load_state_dict(torch.load(model_name, weights_only=True))
    
    running_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            X, theta_true= data[0].to(device), data[1].to(device)
            theta_pred = model(X)
            loss = test_func(theta_pred, theta_true)
            running_loss += loss.item()

        running_loss /= len(test_loader)

    return running_loss


# class CNN(nn.Module):

#     def __init__(self, M: int, D:int, device=dev):

#         super().__init__()
#         self.m = M
#         self.d = D
#         self.net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 3), stride=2, device=device),
#                                  nn.BatchNorm2d(256, device=device), nn.ReLU(),
#                                 #  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), device=device),
#                                 #  nn.BatchNorm2d(256, device=device), nn.ReLU(),
#                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), device=device),
#                                  nn.BatchNorm2d(256, device=device), nn.ReLU(),
#                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), device=device),
#                                  nn.BatchNorm2d(256, device=device), nn.ReLU(), nn.Flatten(),
#                                  nn.Linear(in_features=256*(int((self.m - 3)/2)-1), out_features=4096, device=device),
#                                  nn.ReLU(), nn.Dropout(0.3),
#                                  nn.Linear(in_features=4096, out_features=2048, device=device),
#                                  nn.ReLU(), nn.Dropout(0.3),
#                                  nn.Linear(in_features=2048, out_features=1024, device=device),
#                                  nn.ReLU(), nn.Dropout(0.3),
#                                  nn.Linear(in_features=1024, out_features=nbSamples_cnn, device=device),
#                                  nn.Sigmoid())
#         self.device = device
        

#     def forward(self, x):

#         Rx = x.transpose(1, 2) @ x.conj() / x.shape[1]

#         X = torch.stack((torch.real(Rx),
#                          torch.imag(Rx),
#                          torch.angle(Rx)), dim=1).to(self.device)
        
#         return self.net(X)