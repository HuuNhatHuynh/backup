import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import *

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Kalman:

    def __init__(self, 
                 array: ArrayModel, 
                 d: int,
                 Ts: float,  
                 W: torch.Tensor, 
                 V: torch.Tensor):
        """
        Args:
            array (ArrayModel): The array model under consideration.
            d (int): The number of sources to be estimated.
            Ts (float): The sampling time interval used in the Kalman filter.
            W (torch.Tensor): The covariance matrix of the process noise, with dimensions (d, 2, 2).
            V (torch.Tensor): The covariance matrix of the measurement noise, with dimenstions (d)

        Returns:
            None
        """
        self.array = array
        self.m = self.array.m
        self.d = d
        self.Ts = Ts
        self.F = torch.Tensor([[[1, Ts],
                                [0, 1]]]).repeat(d, 1, 1)
        self.H = torch.Tensor([[[1, 0]]]).repeat(d, 1, 1)
        self.W = W
        self.V = V

    def initialize(self, X: torch.Tensor):
        """
        Args:
            X (torch.Tensor): observations collected during first 2Ts seconds, with dimensions (2*Ns, m)

        Returns:
            
        """
        cov_0 = compute_covariance(X[:int(X.shape[0]/2)])
        theta_0, S, sigma2 = ESPRIT(Rx=cov_0, d=self.d, array=self.array)
        cov_1 = compute_covariance(X[int(X.shape[0]/2):])
        theta_1, _, _      = ESPRIT(Rx=cov_1, d=self.d, array=self.array)

        self.S = S
        self.sigma2 = sigma2

        self.x = torch.stack((theta_1, (theta_1 - theta_0) / self.Ts), dim=1).unsqueeze(2)
        self.P = torch.stack((torch.stack((self.V, self.V / self.Ts), dim=1),
                              torch.stack((self.V / self.Ts, 2 * self.V / self.Ts ** 2), dim=1)), dim=1)
        
        return theta_0, theta_1
        
    def estimate(self, X: torch.Tensor, modified: bool):
        """
        Args:
            X (torch.Tensor): observations collected during previous Ts seconds, with dimensions (Ns, m)

        Returns:
        """
        x_pred = self.F @ self.x
        theta_pred = (self.H @ x_pred).squeeze()
        A_pred = self.array.get_steering_vector(theta=theta_pred)
        R_pred = A_pred @ self.S @ A_pred.T.conj() + self.sigma2 * torch.eye(self.m)
        R_x = compute_covariance(X)

        if modified:
            innov = compute_angle_displacement_ver3(new_cov=R_x, theta=theta_pred, array=self.array)
        else:
            innov = compute_angle_displacement(delta_cov=R_x-R_pred, theta=theta_pred, S=self.S, array=self.array)
        
        innov = innov.reshape(self.d, 1, 1)

        P_pred = self.F @ self.P @ self.F.transpose(1, 2) + self.W

        R_innov = self.H @ P_pred @ self.H.transpose(1, 2) + self.V.reshape(self.d, 1, 1)
        G = (P_pred @ self.H.transpose(1, 2)) / R_innov
        # G = (P_pred @ self.H.transpose(1, 2)) @ torch.linalg.inv(R_innov)
        self.x = x_pred + G @ innov
        self.P = (torch.eye(2) - G @ self.H) @ P_pred

        theta_estimated = (self.H @ self.x).squeeze()
        
        return theta_estimated
    


def compute_angle_displacement(delta_cov: torch.Tensor, theta: torch.Tensor, S: torch.Tensor, array: ArrayModel) -> torch.Tensor:
    """
    Compute the displacement of angles given the change in observed covariance and previous angles.

    Args:
        delta_cov (torch.Tensor): Change in observed covariance, provided in batches.
        theta (torch.Tensor): Previous angles, provided in batches.
        S (torch.Tensor): Source covariance matrix, provided in batches.
        array (ArrayModel): The array model to use for computation.

    Returns:
        torch.Tensor: Displacement of angles.
    """
    A = array.get_steering_vector(theta)
    J = array.get_first_derivative_steering_vector(theta)
    u = (A @ S)[..., 0, :].conj().unsqueeze(-2)
    B = J[..., 1:, :] * u
    y = delta_cov[..., 1:, 0]

    B = torch.cat((torch.real(B), torch.imag(B)), dim=-2)
    y = torch.cat((torch.real(y), torch.imag(y)), dim=-1)
    dis = torch.linalg.lstsq(B, y).solution
    print(torch.linalg.norm(dis))

    # dis = torch.linalg.inv(B.T.conj() @ B + torch.diag(torch.diag(B.T.conj() @ B)) / 20) @ B.T.conj() @ y
    # dis = torch.real(dis)

    return dis



# def compute_angle_displacement_ver_2(delta_cov: torch.Tensor, theta: torch.Tensor, s: torch.Tensor, array: ArrayModel):
#     """
#     Compute the displacement of angles and source powers given the change in observed covariance and previous angles.

#     Args:
#         delta_cov (torch.Tensor): Change in observed covariance, provided in batches.
#         theta (torch.Tensor): Previous angles, provided in batches.
#         s (torch.Tensor): Source power vector, provided in batches.
#         array (ArrayModel): The array model to use for computation.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Displacement of angles and source powers.
#     """
#     d = theta.shape[-1]
#     A = array.get_steering_vector(theta)
#     J = array.get_first_derivative_steering_vector(theta)
#     u = (A * s.unsqueeze(-2))[..., 0, :].conj().unsqueeze(1)
#     M = torch.cat((J * u, A), dim=-1)
#     n = delta_cov[:, :, 0]
#     dis = torch.linalg.lstsq(torch.cat((torch.real(M), torch.imag(M)), dim=-2),
#                              torch.cat((torch.real(n), torch.imag(n)), dim=-1)).solution
#     dis_theta, dis_s = dis[..., :d], dis[..., d:]
        
#     return dis_theta, dis_s



def compute_angle_displacement_ver3(new_cov: torch.Tensor, theta: torch.Tensor, array: ArrayModel):

    m = array.m 
    d = theta.shape[-1]
    _, vecs = torch.linalg.eigh(new_cov)
    noise_space = vecs[..., :(m-d)]
    proj_noise_space = noise_space @ noise_space.conj().transpose(-2, -1)
    A = array.get_steering_vector(theta=theta)
    dA = array.get_first_derivative_steering_vector(theta=theta)
    num = torch.diagonal(A.conj().transpose(-2, -1) @ proj_noise_space @ dA \
                       + dA.conj().transpose(-2, -1) @ proj_noise_space @ A, dim1=-2, dim2=-1)
    den = - 2 * torch.diagonal(dA.conj().transpose(-2, -1) @ proj_noise_space @ dA, dim1=-2, dim2=-1)
    num = torch.real(num)
    den = torch.real(den)
    return num / den