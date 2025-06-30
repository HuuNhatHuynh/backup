import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from SubspaceTracker import *


class KalmanNet(nn.Module):

    def __init__(self,
                 array: ArrayModel,
                 d: int,
                 Ts: float,
                 Ns: int, 
                 device: str) -> None:
        """
        Args:
            array (ArrayModel): The array model under consideration.
            d (int): The number of sources to be estimated.
            Ts (float): The sampling time interval used in the Kalman filter.
            Q (torch.Tensor): The covariance matrix of the process noise, with dimensions (d, 3, 1).
            theta_var_init (torch.Tensor): The initial variance of the angles, with a size of (d).

        Returns:
            None
        """
        super(KalmanNet, self).__init__()
        
        self.array = array

        self.m = self.array.m
        self.d = d
        self.Ts = Ts
        self.Ns = Ns

        self.F = torch.Tensor([[[1, Ts, Ts ** 2 / 2],
                                [0, 1, Ts],
                                [0, 0, 1]]]).repeat(d, 1, 1).to(device)
        
        self.H = torch.Tensor([[[1, 0, 0]]]).repeat(d, 1, 1).to(device)

        self.array.array = self.array.array.to(device)
        self.device = device

        self.fc1 = nn.Linear(2 * self.m * self.d, 50, device=device)
        self.gru = nn.GRU(50, 2 * self.m * self.d, batch_first=True, device=device)
        self.fc2 = nn.Linear(2 * self.m * self.d, 3 * self.d, device=device)

    def initialize(self, X: torch.Tensor):
        """
        Args:
            X (torch.Tensor): observations collected during first 2Ts seconds, with dimensions (2*Ns, m)

        Returns:
            
        """
        cov_0 = compute_covariance(X[:, :self.Ns, :])
        theta_0, _, _ = DoA_Estimation_batch(Rx=cov_0, d=self.d, array=self.array, solver=ESPRIT)
        cov_1 = compute_covariance(X[:, self.Ns:, :])
        theta_1, _, _ = DoA_Estimation_batch(Rx=cov_1, d=self.d, array=self.array, solver=ESPRIT)

        self.x = torch.stack((theta_1, 
                              (theta_1 - theta_0) / self.Ts, 
                              torch.zeros_like(theta_0)), dim=2).unsqueeze(3).to(self.device)

        beta = 0.95
        P = 0.01 * torch.eye(self.d).unsqueeze(0).repeat(X.shape[0], 1, 1)
        P = P.to(torch.complex64).to(self.device)
        W = torch.cat((torch.eye(self.d), torch.zeros(self.m-self.d, self.d)), dim=0).unsqueeze(0).repeat(X.shape[0], 1, 1)
        W = W.to(torch.complex64).to(self.device)

        self.tracker = OPAST(beta, P, W)

        return theta_0, theta_1


    def estimate(self, X: torch.Tensor):
        """
        Args:
            X (torch.Tensor): observations collected during previous Ts seconds, with dimensions (Ns, m)

        Returns:
        """
        x_pred = self.F @ self.x
        theta_pred = (self.H @ x_pred).squeeze()
        h = self.array.get_steering_vector(theta_pred)
        h = torch.flatten(h, start_dim=-2, end_dim=-1)
        h = torch.cat((torch.real(h), torch.imag(h)), dim=-1).unsqueeze(0)

        input = self.tracker.feed(X)
        input = torch.flatten(input, start_dim=-2,end_dim=-1)
        input = torch.cat((torch.real(input), torch.imag(input)), dim=-1)

        input = self.fc1(input)
        _, delta = self.gru(input, h)
        delta = delta.squeeze(0)
        delta = self.fc2(delta)
        delta = torch.unflatten(delta, dim=-1, sizes=(self.d, 3, 1))

        self.x = x_pred + delta 
        theta_estimated = (self.H @ self.x).squeeze()
        
        return theta_estimated
    
    def forward(self, X: torch.Tensor):

        self.initialize(X[:, :2 * self.Ns, :])
        y = []
        for i in range(2, int(X.shape[1]/self.Ns)):
            y.append(self.estimate(X[:, self.Ns * i : self.Ns * (i + 1)]))
        y = torch.stack(y, dim=-2)

        return y
    


# class KalmanGain(nn.Module):

#     def __init__(self, m: int, d:int, input_gru: int, hidden_gru: int, device: str) -> None:
        
#         super(KalmanGain, self).__init__()
#         self.m = m
#         self.d = d
#         self.input_gru = input_gru
#         self.hidden_gru = hidden_gru
#         self.device = device
#         self.fc1 = nn.Linear(in_features=2*m, out_features=input_gru, device=device)
#         self.gru = nn.GRU(input_size=input_gru, hidden_size=hidden_gru, batch_first=True, device=device)
#         self.fc2 = nn.Linear(in_features=hidden_gru, out_features=3*d, device=device)

#     def forward(self, X: torch.Tensor):

#         X = torch.cat((torch.real(X), torch.imag(X)), dim=-1)
#         X = self.fc1(X)
#         _, h = self.gru(X)
#         y = self.fc2(h.squeeze(0))
#         y = torch.unflatten(y, dim=-1, sizes=(self.d, 3, 1))
#         return y

    

# class KalmanGain(nn.Module):

#     def __init__(self, 
#                  d: int, m: int,
#                  hidden_size: int, mult: int,
#                  device: str) -> None:
    
#         super(KalmanGain, self).__init__()

#         self.d = d
#         self.m = m
#         self.hidden_size = hidden_size
#         self.device = device

#         self.fc1 = nn.ModuleList([nn.Linear(in_features=m, out_features=mult*m, device=device) for _ in range(d)])
#         self.gru = nn.ModuleList([nn.GRU(input_size=mult*m, hidden_size=hidden_size, batch_first=True, device=device) for _ in range(d)])
#         self.fc2 = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=m, device=device) for _ in range(d)])

#     def initialize(self, h: torch.Tensor) -> None:

#         self.h = h.to(self.device)

#     def forward(self, del_x_hat: torch.Tensor, del_y: torch.Tensor) -> torch.Tensor:

#         list_h = []
#         list_K = []

#         for i in range(self.d):

#             y = torch.cat((del_x_hat[..., i, :], del_y[..., i, :]), dim=-1)
#             y = self.fc1[i](y)
#             y = F.normalize(y, p=2.0, dim=-1, eps=1e-12)
#             _, y = self.gru[i](y.unsqueeze(-2), self.h[..., i, :].unsqueeze(0).contiguous())
#             y = y.squeeze(0)
#             list_h.append(y)
#             K = self.fc2[i](y)
#             list_K.append(K)

#         self.h = torch.stack(list_h, dim=-2)
#         K = torch.stack(list_K, dim=-2)
#         K = torch.unflatten(K, dim=-1, sizes=(self.m, self.n))

#         return K



def compute_innovation(W: torch.Tensor,
                       theta: torch.Tensor,
                       array: ArrayModel,
                       second_order_term=False):
    """
    Args:
        U (torch.Tensor): signal subspace, tensor of shape (..., m, d)
        theta (torch.Tensor): angles of arrival, tensor of shape (..., d)
        array (ArrayModel): considered array model
        second_order_term (bool): use second order term or not

    Returns:
        torch.Tensor: tensor of shape (..., d)
    """
    I = torch.eye(array.m, device=theta.device)
    P = I - W @ W.conj().transpose(-2, -1)

    A = array.get_steering_vector(theta)
    dA = array.get_first_derivative_steering_vector(theta)

    num = torch.real(dA.conj().transpose(-2, -1) @ P @ A)
    
    den = torch.real(dA.conj().transpose(-2, -1) @ P @ dA)
    if second_order_term:
        ddA = array.get_second_derivative_steering_vector(theta)
        den = den + torch.real(ddA.conj().transpose(-2, -1) @ P @ A)
    
    num = torch.diagonal(num, offset=0, dim1=-2, dim2=-1)
    den = torch.diagonal(den, offset=0, dim1=-2, dim2=-1)

    return - num / den