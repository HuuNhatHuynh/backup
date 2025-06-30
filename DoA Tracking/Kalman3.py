import torch 
from math import *
from utils import *
from abc import ABC, abstractmethod



class KalmanFilter(ABC):

    def __init__(self,
                 array: ArrayModel,
                 d: int,
                 Ts: float, 
                 alpha: torch.Tensor, 
                 x_init: torch.Tensor,
                 P_init: torch.Tensor,
                 Q: torch.Tensor,
                 R: torch.Tensor) -> None:
        
        self.array = array
        self.m = self.array.m
        self.d = d
        self.Ts = Ts

        self.F = torch.cat((torch.cat((torch.kron(torch.Tensor([[1, Ts], [0, 1]]), torch.eye(d)), 
                                       torch.zeros(2*d, 2*d)), dim=1), 
                            torch.cat((torch.zeros(2*d, 2*d), 
                                       torch.cat((torch.cat((torch.diag(torch.real(alpha)), - torch.diag(torch.imag(alpha))), dim=1), 
                                                  torch.cat((torch.diag(torch.imag(alpha)), torch.diag(torch.real(alpha))), dim=1)), dim=0)), dim=1)), dim=0)
        
        self.Q = Q
        self.R = R

        self.x = x_init
        self.P = P_init

    @abstractmethod
    def step(self, y: torch.Tensor):
        pass



class ExtendedKalmanFilter(KalmanFilter):

    def step(self, y: torch.Tensor):

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y_pred, Jh = h_ekf(x_pred, self.array)
        S = Jh @ P_pred @ Jh.T + self.R 
        KG = P_pred @ Jh.T @ torch.linalg.inv(S)
        self.x = x_pred + KG @ (y - y_pred)
        self.P = (torch.eye(4*self.d) - KG @ Jh) @ P_pred

        return self.x[:self.d]
    


class UnscentedKalmanFilter(KalmanFilter):

    def step(self, y: torch.Tensor):

        kappa = 0
        alpha = 0.3
        beta = 2
        
        n = 4 * self.d
        lamda = alpha ** 2 * (n + kappa) - n

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        L = torch.linalg.cholesky(P_pred)
        
        sigma_points = torch.cat((x_pred.unsqueeze(1), 
                                  x_pred.unsqueeze(1) + sqrt(n + lamda) * L, 
                                  x_pred.unsqueeze(1) - sqrt(n + lamda) * L), dim=1)
        
        weights_mean = torch.full((2 * n + 1, ), 1 / (2 * (n + lamda)))
        weights_cov = torch.clone(weights_mean)
        weights_mean[0] = lamda / (n + lamda)
        weights_cov[0] = lamda / (n + lamda) + 1 - alpha ** 2 + beta
        
        sigma_points_transformed = h_ukf(sigma_points, self.array)
        
        y_pred = torch.einsum('ij,j->i', sigma_points_transformed, weights_mean)
        
        S = (sigma_points_transformed - y_pred.unsqueeze(1)) \
          @ torch.diag(weights_cov) \
          @ (sigma_points_transformed - y_pred.unsqueeze(1)).T + self.R
        
        C = (sigma_points - x_pred.unsqueeze(1)) \
          @ torch.diag(weights_cov) \
          @ (sigma_points_transformed - y_pred.unsqueeze(1)).T
        
        KG = C @ torch.linalg.inv(S)
        
        self.x = x_pred + torch.einsum('ij,j->i', KG, y - y_pred)
        self.P = P_pred - KG @ S @ KG.T

        return self.x[:self.d]
    


class SquareRoot_UnscentedKalmanFilter(KalmanFilter):

    def step(self, y: torch.Tensor):

        pass



def h_ekf(x: torch.Tensor, array: ArrayModel):

    d = int(x.shape[0] / 4)
    m = array.m
    theta = x[:d]

    A = array.get_steering_vector(theta)
    A = torch.cat((torch.cat((torch.real(A), - torch.imag(A)), dim=1), 
                   torch.cat((torch.imag(A), torch.real(A)), dim=1)), dim=0)
    y = A @ x[2*d:]
    
    dA = array.get_first_derivative_steering_vector(theta)
    dA = torch.cat((torch.cat((torch.real(dA), - torch.imag(dA)), dim=1), 
                    torch.cat((torch.imag(dA), torch.real(dA)), dim=1)), dim=0)
    L = dA @ torch.cat((torch.diag(x[2*d:3*d]), 
                        torch.diag(x[3*d:])), dim=0)

    Jh = torch.cat((L, torch.zeros(2*m, d), A), dim=1)
    
    return y, Jh


def h_ukf(x: torch.Tensor, array: ArrayModel):

    d = int(x.shape[0] / 4)
    theta = x[:d]

    A = array.get_steering_vector(theta.T)
    A = torch.cat((torch.cat((torch.real(A), - torch.imag(A)), dim=2), 
                   torch.cat((torch.imag(A), torch.real(A)), dim=2)), dim=1)
    
    y = torch.einsum('ijk,ki->ji', A, x[2*d:, :])
    
    return y