import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import find_peaks
from math import sqrt, factorial
from itertools import permutations
from random import random
from ArrayModel import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DATASET(Dataset):

    def __init__(self, observations: torch.Tensor, angles: torch.Tensor):
        
        self.observations = observations
        self.angles = angles

    def __getitem__(self, idx):

        return self.observations[idx], self.angles[idx]
    
    def __len__(self):

        return len(self.observations)
    


class RMSPE(nn.Module):

    def __init__(self, d: int, device:str):

        super().__init__()

        self.d = d
        self.permutation_matrices = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)
        self.permutation_matrices = self.permutation_matrices.to(device)

    def forward(self, angle_pred: torch.Tensor, angle_true: torch.Tensor):

        angle_true_perms = torch.einsum('pmn,bn->bmp', self.permutation_matrices, angle_true)
        diff = angle_true_perms - angle_pred.unsqueeze(-1)
        diff = torch.remainder(diff + torch.pi / 2, torch.pi) - torch.pi / 2
        # diff = torch.fmod(diff + torch.pi / 2, torch.pi) - torch.pi / 2
        loss = torch.mean(torch.amin(torch.mean(diff**2+1e-10, dim=1) ** (1/2), dim=1))

        return loss 


class MSPE(nn.Module):

    def __init__(self, d: int, device: str = 'cpu'):

        super().__init__()
        self.d = d
        self.permutation_matrices = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)
        self.permutation_matrices = self.permutation_matrices.to(device)

    def forward(self, angle_pred: torch.Tensor, angle_true: torch.Tensor):

        angle_true_perms = torch.einsum('pmn,bn->bmp', self.permutation_matrices, angle_true)
        diff = angle_true_perms - angle_pred.unsqueeze(-1)
        diff = (diff + torch.pi / 2) % torch.pi - torch.pi / 2
        # diff = torch.fmod(diff + torch.pi / 2, torch.pi) - torch.pi / 2
        loss = torch.mean(torch.amin(torch.mean(diff**2, dim=1), dim=1))
        return loss
    


class RMSPE_test:

    def __init__(self, d: int, modulo: bool = True, device: str = 'cpu'):
        
        self.d = d
        self.modulo = modulo
        self.permutation_matrices = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)
        self.permutation_matrices = self.permutation_matrices.to(device)

    def calculate(self, angle_pred, angle_true):

        angle_true_perms = torch.einsum('pmn,bn->bmp', self.permutation_matrices, angle_true)
        diff = angle_true_perms - angle_pred.unsqueeze(-1)
        if self.modulo:
            diff = (diff + torch.pi/2) % torch.pi - torch.pi/2
        losses = torch.amin(torch.mean(diff**2, dim=1) ** (1/2), dim=1)

        return losses
    


# def no_regularization(x: torch.Tensor):
    
#     return 0



# def regularization_sigmoid(x: torch.Tensor, v_min: float = -torch.pi/2-0.3, v_max: float = torch.pi/2+0.3, alpha: float = 20.0):

#     y = 1 / (1 + torch.exp(alpha*(x - v_min))) + 1 / (1 + torch.exp(-alpha*(x-v_max)))
#     return torch.sum(y)



# def regularization_polynomial(x: torch.Tensor, v_min: float = -torch.pi/2, v_max: float = torch.pi/2, alpha: float = 2.0, k: int = 2):
    
#     y = torch.where(x < v_min, (alpha*(v_min - x)) ** k, 0) + torch.where(x > v_max, (alpha*(x - v_max)) ** k, 0)
#     return torch.sum(y)



def generate_direction(d: int, gap: float, angle_min: float, angle_max: float) -> torch.Tensor:
    """
    Generates `D` random directions, ensuring a minimum angular separation.

    Parameters:
    D (int): Number of source directions to generate.
    gap (float): Minimum angular separation between consecutive directions (default is 0.1 radians).

    Returns:
    torch.Tensor: A sorted tensor of size (D) representing the generated directions in radians.
    """

    if d == 1:
        theta = torch.rand(1) * (angle_max - angle_min) + angle_min

    else:
        while True:
            theta = torch.rand(d) * (angle_max - angle_min) + angle_min
            theta = theta.sort()[0]  
            if torch.min(theta[1:] - theta[:-1]) > gap and theta[-1] - theta[0] < torch.pi - gap:
                break 
    
    return theta



def generate_qpsk_symbols(nbSources: int, nbSymbols: int):
    
    bits = torch.randint(0, 2, (nbSources, 2 * nbSymbols))
    real_part = 2 * bits[:, 0::2] - 1  
    imag_part = 2 * bits[:, 1::2] - 1  
    qpsk_symbols = (real_part + 1j * imag_part) / sqrt(2)
    
    return qpsk_symbols



def generate_signal(t: int, d: int, snr: float, array, qpsk: bool, coherent: bool, gap: float, angle_min: float, angle_max: float):
    """
    Generates an array signal with specified characteristics.

    Parameters:
    T (int): Length of the generated signals (number of time samples).
    D (int): Number of signal sources.
    SNR (float): Signal-to-noise ratio (in dB).
    array (ArrayModel): Predefined array model used to calculate steering vectors.
    coherent (bool): If True, the generated signals will be coherent. Otherwise, they will be independent (default is False).

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        - Signal matrix of size (M, T), where M is the number of array elements.
        - Vector of angles of size (D), representing the directions of arrival (DOAs) for each source.
        - perturbation of sensors in x directions
        - perturbation of sensors in y directions
    """

    theta = generate_direction(d, gap, angle_min, angle_max)

    A = array.get_steering_vector(theta)
    
    if not coherent:
        if qpsk:
            x = generate_qpsk_symbols(d, t) 
        else:
            x = (torch.randn(d, t) + 1j * torch.randn(d, t)) / sqrt(2)
    else:
        if qpsk:
            x = generate_qpsk_symbols(1, t) 
        else:
            x = (torch.randn(1, t) + 1j * torch.randn(1, t)) / sqrt(2)
        x = torch.Tensor.repeat(x, (d, 1))

    n = (torch.randn(array.m, t) + 1j * torch.randn(array.m, t)) / sqrt(2) / 10 ** (snr / 20) 

    return A @ x + n, theta



def generate_data(n: int, t: int, d: int, snr_min: float, snr_max: float, array, qpsk: bool, coherent: bool, 
                  gap: float = 0.1, angle_min: float = -torch.pi/2, angle_max: float = torch.pi/2):
    """
    Generates a dataset of array signals and corresponding incident angles.

    Parameters:
    N (int): Number of signal samples to generate.
    T (int): Length of each signal (number of time samples).
    D (int): Number of sources (directions of arrival).
    SNR (float): Signal-to-noise ratio (in dB).
    array (ArrayModel): Predefined array model used to calculate steering vectors.
    coherent (bool): If True, the generated signals will be coherent across sources (default is False).

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        - `observations`: Tensor of size (N, T, M) representing the array signals, where M is the number of array elements.
        - `angles`: Tensor of size (N, D) containing the incident angles (DOAs) for each sample.
    """
    
    observations = []
    angles = []
    
    for _ in range(n):
        snr = random() * (snr_max - snr_min) + snr_min
        y, theta = generate_signal(t, d, snr, array, qpsk, coherent, gap, angle_min, angle_max)
        observations.append(y.T) 
        angles.append(theta) 
    
    observations = torch.stack(observations, dim=0)
    angles = torch.stack(angles, dim=0)
    
    return observations, angles



# def get_spectrum(En: torch.Tensor, array: ArrayModelAbstractClass) -> torch.Tensor:
#     """
#     Calculate the MUSIC pseudospectrum based on the given noise subspace and array model.

#     Parameters:
#     En : torch.Tensor
#         The noise subspace, which can have dimensions (M, M-D) or (N, M, M-D). 
#         Here, M is the number of sensors, D is the number of signals, and N is the number of samples.

#     array : ArrayModel
#         An instance of the ArrayModel that defines the array configuration and properties.

#     Returns:
#     torch.Tensor
#         The computed pseudospectrum. It has dimensions (nbSamples_spectrum) 
#         for a single sample or (N, nbSamples_spectrum) if multiple samples are processed.
#     """
#     array.x = array.x.to(En.device)
#     array.y = array.y.to(En.device)
#     A = array.array_manifold.to(En.device)
#     spectrum = 1 / torch.norm(En.conj().transpose(-2, -1) @ A, dim=-2) ** 2
    
#     return spectrum



def find_peaks_customized(spectrum: torch.Tensor):

    idx, _ = find_peaks(spectrum)
    if spectrum[0] > spectrum[1]:
        idx = np.insert(idx, 0, 0)
    if spectrum[-2] < spectrum[-1]:
        idx = np.append(idx, spectrum.shape[0] - 1)
    return idx



def MUSIC(X: torch.Tensor, d: int, array: ArrayModelAbstractClass):

    m, t = X.shape
    cov = X @ X.T.conj() / t
    vals, vecs = torch.linalg.eigh(cov)
    vecs = vecs[:, vals.sort()[1]]
    En = vecs [:, :(m-d)]
    spectrum = 1 / torch.norm(En.conj().transpose(-2, -1) @ array.array_manifold, dim=-2) ** 2
    idx = find_peaks_customized(spectrum)
    idx = idx[list(torch.argsort(spectrum[idx])[-d:])]

    if len(idx) < d:

        theta = torch.cat((array.angles_spectrum[idx], torch.pi * (torch.rand(d - len(idx)) - 1/2)))
        return theta, spectrum, False

    theta = array.angles_spectrum[idx]
    return theta, spectrum, True



def Root_MUSIC(X, d, array: ArrayModelAbstractClass):

    _, t, m = X.shape
    lamda = array.lamda
    distance = sqrt((array.x[1] - array.x[0]) ** 2 + (array.y[1] - array.y[0]) ** 2)

    Rx = X.transpose(1, 2) @ X.conj() / t

    vals, vecs = torch.linalg.eigh(Rx)
    idx = torch.sort(vals, dim=1)[1].unsqueeze(dim=1).repeat(repeats=(1, m, 1))
    vecs = torch.gather(vecs, dim=2, index=idx)
    En = vecs[:, :, :(m - d)]

    R = En @ En.conj().transpose(-2, -1)
    coef = torch.zeros(Rx.shape[0], 2 * m - 1, dtype=torch.complex64, device=Rx.device)
    for i in range(2 * m - 1):
        coef[:, i] = torch.diagonal(R, offset=i - m + 1, dim1=-2, dim2=-1).sum(dim=-1)
    coef = coef / coef[:, -1].unsqueeze(1)
    C = torch.zeros(Rx.shape[0], 2 * m - 2, 2 * m - 2, dtype=torch.complex64, device=Rx.device)
    C[:, :, -1] = - coef[:, :-1]
    C[:, 1:, :-1] = torch.eye(2 * m - 3, dtype=torch.complex64, device=Rx.device)
    roots = torch.linalg.eigvals(C)

    roots = roots[torch.abs(roots) < 1]
    roots = roots.reshape(-1, m - 1)

    indx_sorted = torch.argsort(torch.abs(torch.abs(roots) - 1), dim=1)
    roots = torch.gather(roots, dim=1, index=indx_sorted)[:, :d]
    
    return - torch.arcsin(torch.angle(roots) * lamda / (2 * torch.pi * distance))



# def permutations(arr):
#     """
#     Recursively generates all possible permutations of a given list.
    
#     Args:
#     arr (list): A list of elements for which to compute permutations.

#     Returns:
#     list: A list containing all permutations of the input list, 
#           where each permutation is also a list.
#     """
#     if len(arr) == 1:
#         return [arr]
    
#     res = []
    
#     for i in range(len(arr)):
#         res += [[arr[i]] + perm for perm in permutations(arr[:i] + arr[i+1:])]
    
#     return res



def CRLB_stochastic(array, angles, true_sources_cov, noise_sigma2, n_snapshots):
    """
    References:
        [1] P. Stoica and A. Nehorai, "Performance study of conditional and
        unconditional direction-of-arrival estimation," IEEE Transactions on
        Acoustics, Speech and Signal Processing, vol. 38, no. 10,
        pp. 1783-1795, Oct. 1990.
    """
    A, D = array.get_steering_vector(angles, True)
    I = torch.eye(array.m, dtype=torch.complex64)
    R = A @ true_sources_cov @ A.T.conj() + noise_sigma2 * I
    H = D.T.conj() @ (I - A @ torch.linalg.inv(A.T.conj() @ A) @ A.T.conj()) @ D
    CRB = torch.real(H * (true_sources_cov @ A.T.conj() @ torch.linalg.inv(R) @ A @ true_sources_cov).T)
    CRB = torch.linalg.inv(CRB) * (noise_sigma2 / n_snapshots / 2)
    
    return (CRB + CRB.T) / 2


def CLRB_deterministic(array, angles, sample_sources_cov, noise_sigma2, n_snapshots):
    """
    References:
        [1] P. Stoica and A. Nehorai, "Performance study of conditional and
        unconditional direction-of-arrival estimation," IEEE Transactions on
        Acoustics, Speech and Signal Processing, vol. 38, no. 10,
        pp. 1783-1795, Oct. 1990.
    """
    A, D = array.get_steering_vector(angles, True)
    I = torch.eye(array.m, dtype=torch.complex64)
    H = D.T.conj() @ (I - A @ torch.linalg.inv(A.T.conj() @ A) @ A.T.conj()) @ D
    CRB = torch.linalg.inv(torch.real(H * sample_sources_cov.T)) * noise_sigma2 / n_snapshots / 2
    return (CRB + CRB.T) / 2