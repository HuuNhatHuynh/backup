import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DA_MUSIC(nn.Module):

    def __init__(self, dmin: int, dmax: int, array, device=dev):
        
        super().__init__()

        self.m = array.m
        self.dmin = dmin 
        self.dmax = dmax

        self.A = array.array_manifold.to(device)

        self.bn = nn.BatchNorm1d(2*self.m, device=device)
        self.rnn = nn.GRU(input_size=2*self.m, hidden_size=2*self.m, num_layers=1, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=2*self.m, out_features=2*self.m*self.m, device=device)

        self.mlp_selection = nn.Sequential(nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(), 
                                           nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(), 
                                           nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(), 
                                           nn.Linear(in_features=2*self.m, out_features=self.m, device=device), nn.Sigmoid())
        
        self.mlp_doa = nn.Sequential(nn.Linear(in_features=array.nbSamples_spectrum, out_features=2*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m, out_features=self.m-1, device=device))
        
        self.mlp_d = nn.Sequential(nn.Linear(in_features=2*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=2*self.m*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=2*self.m*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=2*self.m*self.m, out_features=dmax-dmin+1, device=device), nn.Softmax(dim=1))

    def forward(self, X: torch.Tensor):
        
        X = torch.cat((torch.real(X), torch.imag(X)), dim=-1)
        X = self.bn(X.transpose(1, 2)).transpose(1, 2)
        _, X = self.rnn(X)

        cov = self.fc(X[-1])
        cov = cov.reshape(-1, 2, self.m, self.m)
        cov = cov[:, 0, :, :] + 1j * cov[:, 1, :, :]

        vals, vecs = torch.linalg.eig(cov)
        idx = torch.argsort(torch.abs(vals), dim=1)
        vals = torch.gather(vals, dim=1, index=idx)
        idx = idx.unsqueeze(dim=1).repeat(repeats=(1, self.m, 1))
        vecs = torch.gather(vecs, dim=2, index=idx)

        vals = torch.cat((torch.real(vals), torch.imag(vals)), dim=1)
        
        weight_selection = self.mlp_selection(vals).to(torch.complex64)
        vecs = vecs @ torch.diag_embed(weight_selection)
        spectrum = 1 / torch.norm(vecs.conj().transpose(1, 2) @ self.A, dim=1) ** 2    
        angles = self.mlp_doa(spectrum)

        vals = vals.detach()
        nbSources = self.mlp_d(vals)

        return angles, nbSources
    


class DA_MUSIC_v2(nn.Module):

    def __init__(self, dmin: int, dmax: int, array, device=dev):
        
        super().__init__()

        self.m = array.m
        self.dmin = dmin 
        self.dmax = dmax

        self.A = array.array_manifold.to(device)

        self.bn = nn.BatchNorm1d(2*self.m, device=device)
        self.rnn = nn.GRU(input_size=2*self.m, hidden_size=2*self.m, num_layers=1, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=2*self.m, out_features=2*self.m*self.m, device=device)

        self.mlp_selection = nn.Sequential(nn.Linear(in_features=self.m, out_features=self.m, device=device), nn.ReLU(), 
                                           nn.Linear(in_features=self.m, out_features=self.m, device=device), nn.ReLU(), 
                                           nn.Linear(in_features=self.m, out_features=self.m, device=device), nn.ReLU(), 
                                           nn.Linear(in_features=self.m, out_features=self.m, device=device), nn.Sigmoid())
        
        self.mlp_doa = nn.Sequential(nn.Linear(in_features=array.nbSamples_spectrum, out_features=2*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m, out_features=2*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m, out_features=self.m-1, device=device))
        
        self.mlp_d = nn.Sequential(nn.Linear(in_features=self.m, out_features=self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=self.m*self.m, out_features=self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=self.m*self.m, out_features=self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=self.m*self.m, out_features=dmax-dmin+1, device=device), nn.Softmax(dim=1))

    def forward(self, X: torch.Tensor):
        
        X = torch.cat((torch.real(X), torch.imag(X)), dim=-1)
        X = self.bn(X.transpose(1, 2)).transpose(1, 2)
        _, X = self.rnn(X)

        cov = self.fc(X[-1])
        cov = cov.reshape(-1, 2, self.m, self.m)
        cov = cov[:, 0, :, :] + 1j * cov[:, 1, :, :]

        vals, vecs = torch.linalg.eigh(cov)
        idx = torch.argsort(torch.abs(vals), dim=1)
        vals = torch.gather(vals, dim=1, index=idx)
        idx = idx.unsqueeze(dim=1).repeat(repeats=(1, self.m, 1))
        vecs = torch.gather(vecs, dim=2, index=idx)
        
        weight_selection = self.mlp_selection(vals).to(torch.complex64)
        vecs = vecs @ torch.diag_embed(weight_selection)
        spectrum = 1 / torch.norm(vecs.conj().transpose(1, 2) @ self.A, dim=1) ** 2    
        angles = self.mlp_doa(spectrum)

        vals = vals.detach()
        nbSources = self.mlp_d(vals)

        return angles, nbSources
    


class ECNet(nn.Module):

    def __init__(self, m: int, dmin: int, dmax: int,  
                 nbLayers: int, nbNeurons: int, device: str = dev):
        
        super().__init__()
        self.m = m
        self.layers = nn.ModuleList([nn.Linear(in_features=m, out_features=nbNeurons, device=device), nn.ReLU()])
        for _ in range(nbLayers - 2):
            self.layers.append(nn.Linear(in_features=nbNeurons, out_features=nbNeurons, device=device))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=nbNeurons, out_features=dmax-dmin+1, device=device))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x, transformation: str = None):

        cov = x.transpose(1, 2) @ x.conj() / x.shape[1]
        vals = torch.linalg.eigvalsh(cov)
        
        if transformation == "T1":
            pred = torch.arange(1, self.m + 1) * vals / torch.cumsum(vals, dim=1)

        elif transformation == "T2":
            pred = torch.arange(1, self.m + 1) * torch.cumsum(vals ** 2, dim=1) / torch.cumsum(vals, dim=1) ** 2

        else:
            pred = vals

        for layer in self.layers:
            pred = layer(pred)

        return pred
    


class RNN(nn.Module):
        
    def __init__(self, m: int, dmin: int, dmax: int, device=dev):
        
        super().__init__()
        self.m = m
        self.dmin = dmin 
        self.dmax = dmax

        self.bn = nn.BatchNorm1d(2*self.m, device=device)
        self.rnn = nn.GRU(input_size=2*self.m, hidden_size=2*self.m, num_layers=1, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=2*self.m, out_features=2*self.m*self.m, device=device)
        
        self.mlp_doa = nn.Sequential(nn.Linear(in_features=2*self.m*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(),
                                     nn.Linear(in_features=2*self.m*self.m, out_features=self.m-1, device=device))
        
        self.mlp_d = nn.Sequential(nn.Linear(in_features=2*self.m*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=2*self.m*self.m, out_features=2*self.m*self.m, device=device), nn.ReLU(), 
                                   nn.Linear(in_features=2*self.m*self.m, out_features=dmax-dmin+1, device=device), nn.Softmax(dim=1))

    def forward(self, X: torch.Tensor):
        
        X = torch.cat((torch.real(X), torch.imag(X)), dim=-1)
        X = self.bn(X.transpose(1, 2)).transpose(1, 2)
        _, X = self.rnn(X)
        X = X.squeeze(0)
        y = self.fc(X)
        DoAs = self.mlp_doa(y)
        y = y.detach()
        nbSources = self.mlp_d(y) 

        return DoAs, nbSources