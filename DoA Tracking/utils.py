import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from math import sqrt
from scipy.signal import find_peaks

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nbSamples_spectrum = 1000
angles_spectrum = torch.arange(0, nbSamples_spectrum, 1) * torch.pi / nbSamples_spectrum - torch.pi / 2



class DATA(Dataset):

    def __init__(self, X, y, Ns):
        self.X = X
        self.y = y[:, Ns-1::Ns]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)



class ArrayModel:
    """
    A class representing an antenna array model.

    Args:
        M (int): Number of sensors in the array.
        perturbation (float, optional): Percent perturbation added to sensor positions.

    Methods:
        get_steering_vector(theta): Computes the steering vector for given angles.
        get_steering_vector_derivative(theta): Computes the derivative of the steering vector.
    """

    def __init__(self, m: int, perturbation: float = 0):
        
        self.m = m
        self.array = torch.arange(0, m, 1) + (torch.rand(m) * 2 - 1) * perturbation 
    
    def get_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the steering vector for the given incident angles.

        Args:
            theta (torch.Tensor): Incident angles of size (D) or (B, D).

        Returns:
            torch.Tensor: Steering vector of size (M, D) or (B, M, D).
        """
        if len(theta.shape) == 1:
            return torch.exp(-1j * torch.pi * self.array.reshape(-1, 1) * torch.sin(theta).reshape(1, -1))
        elif len(theta.shape) == 2:
            return torch.exp(-1j * torch.pi * self.array.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1))
        
    def get_first_derivative_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute the first derivative of the steering vector with respect to the incident angles.

        Args:
            theta (torch.Tensor): Incident angles of size (D) or (B, D).

        Returns:
            torch.Tensor: Derivative of the steering vector of size (M, D) or (B, M, D).
        """
        if len(theta.shape) == 1:
            factor = -1j * torch.pi * self.array.reshape(-1, 1) * torch.cos(theta).reshape(1, -1)
        elif len(theta.shape) == 2:
            factor = -1j * torch.pi * self.array.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)
        
        return factor * self.get_steering_vector(theta)
    
    def get_second_derivative_steering_vector(self, theta: torch.Tensor):
        """
        Compute the second derivative of the steering vector with respect to the incident angles.

        Args:
            theta (torch.Tensor): Incident angles of size (D) or (B, D).

        Returns:
            torch.Tensor: Derivative of the steering vector of size (M, D) or (B, M, D).
        """
        if len(theta.shape) == 1:
            factor = 1j * torch.pi * self.array.reshape(-1, 1) * torch.sin(theta).reshape(1, -1) \
                   - (torch.pi * self.array.reshape(-1, 1) * torch.cos(theta).reshape(1, -1)) ** 2
        elif len(theta.shape) == 2:
            factor = 1j * torch.pi * self.array.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1) \
                   - (torch.pi * self.array.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)) ** 2
        
        return factor * self.get_steering_vector(theta)



def MUSIC(Rx: torch.Tensor, d: int, array: ArrayModel):
    """
    return estimated angles and pseudospectrum using MUSIC algorithm
    
    INPUT: 
    Rx: covariance of received signal of size (M, M)
    D: number of sources
    array: predefined array model

    OUTPUT: 
    spectrum: pseudospectrum of size (nbSamples_spectrum) and estimated incident angle of size (D)
    """
    m = Rx.shape[0]

    A = array.get_steering_vector(angles_spectrum)
    
    vals, vecs = torch.linalg.eigh(Rx)
    sigma2 = torch.mean(vals[:(m - d)])
    E = vecs[:, :(m - d)]
    spectrum = 1 / torch.norm(E.T.conj() @ A, dim=0) ** 2

    idx, _ = find_peaks(spectrum)
    idx = idx[list(torch.argsort(spectrum[idx])[-d:])]
    theta = torch.sort(angles_spectrum[idx])[0]

    B = array.get_steering_vector(theta)
    S = torch.linalg.inv(B.T.conj() @ B) @ B.T.conj() @ (Rx - sigma2 * torch.eye(m)) @ B @ torch.linalg.inv(B.T.conj() @ B) 
    
    return theta, S, sigma2



def ESPRIT(Rx: torch.Tensor, d: int, array: ArrayModel):
    
    """
    differentiable ESPRIT function

    INPUT:
    Rx: estimated covariance of received signals
    D: number of directions

    OUTPUT:
    estimated incident angles, signal covariance, noise power
    """
    m = Rx.shape[0]
    vals, vecs = torch.linalg.eigh(Rx)
    sigma2 = torch.mean(vals[:(m - d)])
    Es = vecs[:, (m - d):]
    Phi = torch.linalg.pinv(Es[:-1, :]) @ Es[1:, :]
    vals = torch.linalg.eigvals(Phi)
    theta = - torch.arcsin(torch.angle(vals) / torch.pi)
    theta = torch.sort(theta)[0]

    B = array.get_steering_vector(theta)
    I = torch.eye(m, device=Rx.device)
    S = torch.linalg.inv(B.T.conj() @ B) @ B.T.conj() @ (Rx - sigma2 * I) @ B @ torch.linalg.inv(B.T.conj() @ B) 
    
    return theta, S, sigma2



def DoA_Estimation_batch(Rx: torch.Tensor, d: int, array: ArrayModel, solver):
    
    list_theta = []
    list_S = []
    list_sigma2 = []

    for i in range(Rx.shape[0]):
        theta, S, sigma2 = solver(Rx=Rx[i], d=d, array=array)
        list_theta.append(theta)
        list_S.append(S)
        list_sigma2.append(sigma2)

    return torch.stack(list_theta, dim=0), torch.stack(list_S, dim=0), torch.stack(list_sigma2, dim=0)



def compute_covariance(X: torch.Tensor):
    """
    Computes the covariance matrix of the input data matrix X.

    Args:
        X (torch.Tensor): Input data matrix of shape (..., n, m), where 
                          n is the number of snapshots and m is the number of array elements.

    Returns:
        torch.Tensor: Covariance matrix of X with shape (..., m, m).
    """
    return X.transpose(-2, -1) @ X.conj() / X.shape[-2]



def GrassmannianDistance(X: torch.Tensor, Y: torch.Tensor):
    """
    Calculate Grassmannian distance between (batch) matrices X and Y
    
    Args:
        X (torch.Tensor): tensor of size (m, d) or (B, m, d)
        Y (torch.Tensor): tensor of size (m, d) or (B, m, d)

    Returns:
        torch.Tensor
    """
    U, _ = torch.linalg.qr(X)
    V, _ = torch.linalg.qr(Y)
    _, S, _ = torch.linalg.svd(U.conj().transpose(-2, -1) @ V)
    dist = torch.sum(torch.arccos(S) ** 2)

    return dist



def generate_signal(thetas: torch.Tensor, array: ArrayModel, SNR: float):

    t, d = thetas.shape
    m = array.m
    A = array.get_steering_vector(thetas)
    x = (torch.randn(t, d) + 1j * torch.randn(t, d)) / sqrt(2) * 10 ** (SNR / 20)
    n = (torch.randn(t, m) + 1j * torch.randn(t, m)) / sqrt(2)
    
    return torch.einsum('tmd, td->tm', A, x) + n



# def generate_coef_linear(T: float, val_min: float, val_max: float):

#     b = torch.rand(1) * (val_max - val_min) + val_min
#     a = torch.rand(1) * (val_max - val_min) / T + (val_min - b) / T

#     return torch.cat((a, b))


# def generate_coefs(D: int, T: float, val_min: float, val_max: float, gap=0.1):

#     while True:

#         coefs = [generate_coef_linear(T=T, val_min=val_min, val_max=val_max) for _ in range(D)]
#         coefs = torch.stack(sorted(coefs, key = lambda x: x[-1]), dim=0)
        
#         if torch.min(coefs[1:, -1] - coefs[:-1, -1]) > gap and coefs[-1, -1] - coefs[0, -1] < torch.pi - gap:
#             break 
    
#     return coefs


# def generate_thetas(D: int, T: float, Ts: float, Ns: int):

#     time = torch.arange(0, T, Ts/Ns)
#     coefs = generate_coefs(D=D, T=T, val_min=-torch.pi/2, val_max=torch.pi/2)
#     return coefs[:, 0].reshape(1, -1) * time.reshape(-1, 1) + coefs[:, 1].reshape(1, -1)



def generate_theta(D: int, gap: float=0.1):
    while True:
        theta = torch.rand(D) * torch.pi - torch.pi / 2
        theta = theta.sort()[0]
        if torch.min(theta[1:] - theta[:-1]) > gap and theta[-1] - theta[0] < torch.pi - gap:
            break 
    return theta



def generate_thetas(D: int, T: float, Ts: float, Ns: int):
    time = torch.arange(start=0, end=T, step=Ts/Ns)
    theta_begin = generate_theta(D)
    theta_end = generate_theta(D)
    thetas = (theta_end - theta_begin).unsqueeze(0) * time.unsqueeze(1) / T + theta_begin
    return thetas



def generate_dataset(N: int, D: int, T: float, Ts: float, Ns: int, array: ArrayModel, SNR: float):

    X = []
    y = []

    for _ in range(N):

        thetas = generate_thetas(D=D, T=T, Ts=Ts, Ns=Ns)
        signal = generate_signal(thetas=thetas, array=array, SNR=SNR)

        X.append(signal)
        y.append(thetas)

    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)

    return X, y