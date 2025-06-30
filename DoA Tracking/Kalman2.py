import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from SubspaceTracker import *

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Kalman:

    def __init__(self,
                 array: ArrayModel,
                 d: int,
                 Ts: float, 
                 P: torch.Tensor,
                 Q: torch.Tensor) -> None:
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
        
        self.array = array
        self.m = self.array.m
        self.d = d
        self.Ts = Ts

        self.F = torch.Tensor([[[1, Ts, Ts ** 2 / 2],
                                [0, 1, Ts],
                                [0, 0, 1]]]).repeat(d, 1, 1)
        
        self.H = torch.Tensor([[[1, 0, 0]]]).repeat(d, 1, 1)
        
        self.P = P
        self.Q = Q

    def initialize(self, X: torch.Tensor):
        """
        Args:
            X (torch.Tensor): observations collected during first 2Ts seconds, with dimensions (2*Ns, m)

        Returns:
            
        """
        cov_0 = compute_covariance(X[:int(X.shape[0]/3)])
        theta_0, S, sigma2 = ESPRIT(Rx=cov_0, d=self.d, array=self.array)
        cov_1 = compute_covariance(X[int(X.shape[0]/3):int(X.shape[0]*2/3)])
        theta_1, _, _      = ESPRIT(Rx=cov_1, d=self.d, array=self.array)
        cov_2 = compute_covariance(X[int(X.shape[0]*2/3):X.shape[0]])
        theta_2, _, _      = ESPRIT(Rx=cov_2, d=self.d, array=self.array)

        self.Sinv = torch.linalg.inv(S)
        self.sigma2 = sigma2

        self.x = torch.stack((theta_2, 
                            (theta_2 - theta_1) / self.Ts, 
                            (theta_2 - 2 * theta_1 + theta_0) / self.Ts ** 2), dim=1).unsqueeze(2)

        self.past = OPAST(beta=0.95,
                          P=torch.eye(self.d).to(torch.complex64),
                          W=torch.cat((torch.eye(self.d), torch.zeros(self.m - self.d, self.d)), dim=0).to(torch.complex64))

        return theta_0, theta_1, theta_2


    def estimate(self, X: torch.Tensor):
        """
        Args:
            X (torch.Tensor): observations collected during previous Ts seconds, with dimensions (Ns, m)

        Returns:
        """
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.transpose(1, 2) + self.Q
        theta_pred = (self.H @ x_pred).squeeze()

        U = self.past.feed(X)[-1]
        innov = compute_innovation(U=U, theta=theta_pred, array=self.array, second_order_term=False)
        var = compute_var_MUSIC(theta=theta_pred, N=X.shape[0], Sinv=self.Sinv, sigma2=self.sigma2, array=self.array)
        R_innov = self.H @ P_pred @ self.H.transpose(1, 2) + var.reshape(self.d, 1, 1)

        G = P_pred @ self.H.transpose(1, 2) / R_innov
        # G = P_pred @ self.H.transpose(1, 2) @ torch.linalg.inv(R_innov)

        self.x = x_pred + G @ innov.unsqueeze(-1).unsqueeze(-1)
        self.P = (torch.eye(3) - G @ self.H) @ P_pred

        theta_estimated = (self.H @ self.x).squeeze()
        
        return theta_estimated



def compute_var_MUSIC(theta: torch.Tensor,
                      N: int, 
                      Sinv: torch.Tensor,
                      sigma2: float,
                      array: ArrayModel):
    """
    Args:
        theta (torch.Tensor): tensor of shape (..., d)
        N (int): number of snapshots
        Sinv (torch.Tensor): inversed covariance of sources, with dimensions (d, d)
        sigma2 (float): noise power
        array (ArrayModel): considered array model

    Returns:
        torch.Tensor: tensor of shape (..., d)
    """
    A = array.get_steering_vector(theta)
    dA = array.get_first_derivative_steering_vector(theta)

    inv = torch.linalg.inv(A.conj().transpose(-2, -1) @ A)
    P = torch.eye(array.m) - A @ inv @ A.conj().transpose(-2, -1)
    
    num = sigma2 * torch.diagonal(Sinv + sigma2 * Sinv @ inv @ Sinv, offset=0, dim1=-2, dim2=-1)
    den = 2 * N * torch.diagonal(dA.conj().transpose(-2, -1) @ P @ dA, offset=0, dim1=-2, dim2=-1)

    num = torch.real(num)
    den = torch.real(den)
    
    return num / den 



def compute_innovation(U: torch.Tensor,
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
    P = torch.eye(array.m) - U @ U.conj().transpose(-2, -1)

    A = array.get_steering_vector(theta)
    dA = array.get_first_derivative_steering_vector(theta)
    ddA = array.get_second_derivative_steering_vector(theta)

    num = torch.real(dA.conj().transpose(-2, -1) @ P @ A)
    
    den = torch.real(dA.conj().transpose(-2, -1) @ P @ dA)
    if second_order_term:
        den = den + torch.real(ddA.conj().transpose(-2, -1) @ P @ A)
    
    num = torch.diagonal(num, offset=0, dim1=-2, dim2=-1)
    den = torch.diagonal(den, offset=0, dim1=-2, dim2=-1)

    return - num / den