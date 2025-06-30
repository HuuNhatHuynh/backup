import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

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
    


def train(model, nbEpoches, lr, wd, train_loader, valid_loader, model_name, train_func, validation_func, device=dev):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    Loss, Val = [], []
    bestVal = 1000

    for i in tqdm(range(nbEpoches)):
        # Train
        running_loss = 0.0
        for data in train_loader:
            X, theta_true = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            theta_pred = model(X)
            loss = train_func(theta_pred, theta_true) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        Loss.append(running_loss/len(train_loader))

        scheduler.step()

        # Validation 
        with torch.no_grad():
            running_loss = 0.0
            for data in valid_loader:
                X, theta_true = data[0].to(device), data[1].to(device)
                theta_pred = model(X)
                loss = validation_func(theta_pred, theta_true) 
                running_loss += loss.item()
            
            Val.append(running_loss/len(valid_loader))

            if Val[i] < bestVal:
                bestVal = Val[i]
                torch.save(model.state_dict(), model_name)

        print("iteration {}: loss training = {}, best validation = {}".format(i, Loss[-1], bestVal))

    return Loss, Val



def test(model, test_loader, model_name, test_func, device):
    
    if model_name is not None:
        model.load_state_dict(torch.load(model_name, weights_only=True))

    running_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            X, theta_true = data[0].to(device), data[1].to(device)
            theta_pred = model(X)
            loss = test_func(theta_pred, theta_true)
            running_loss += loss.item()

    return running_loss/len(test_loader)



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


# class SoftplusNewtonImplicitLayer(nn.Module):
#     def __init__(self, out_features, device, tol = 1e-4, max_iter=100):
#         super().__init__()
#         self.linear = nn.Linear(out_features, out_features, bias=False, device=device)
#         self.tol = tol
#         self.max_iter = max_iter
#         self.device = device
  
#     def forward(self, x):
#         # Run Newton's method outside of the autograd framework
#         with torch.no_grad():
#             z = torch.tanh(x)
#             self.iterations = 0
#             while self.iterations < self.max_iter:
#                 z_linear = self.linear(z) + x
#                 g = z - torch.tanh(z_linear)
#                 self.err = torch.norm(g)
#                 if self.err < self.tol:
#                     break

#                 # newton step
#                 J = torch.eye(z.shape[1], device=self.device)[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:]
#                 z = z - torch.linalg.solve(J, g[:,:,None])[:,:,0]
#                 self.iterations += 1
    
#         # reengage autograd and add the gradient hook
#         z = nn.Softplus()(self.linear(z) + x)
#         z.requires_grad_()
#         z.register_hook(lambda grad : torch.linalg.solve(J.transpose(1,2), grad[:,:,None])[:,:,0])
#         return z
    

# class RNN2(nn.Module): 
    
#     def __init__(self, m: int, d: int, device=dev):
        
#         super().__init__()

#         self.m = m
#         self.d = d
#         self.device = device

#         self.bn = nn.BatchNorm1d(2*m, device=device)
#         self.rnn = nn.GRU(input_size=2*m, hidden_size=2*m, num_layers=1, batch_first=True, device=device)

#         self.mlp = nn.Sequential(nn.Linear(in_features=2*m, out_features=self.d, device=device), nn.ReLU(),
#                                  SoftplusNewtonImplicitLayer(out_features=self.d, device=device),
#                                  nn.Linear(in_features=self.d, out_features=self.d, device=device))

#     def forward(self, x):
        
#         x = torch.cat((torch.real(x), torch.imag(x)), dim=-1)
#         x = self.bn(x.transpose(1, 2)).transpose(1, 2)
#         _, x = self.rnn(x)
#         y = self.mlp(x[-1])

#         return y