import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import find_peaks
from math import sqrt, factorial
from itertools import permutations
from random import random
import numpy as np
from ArrayModel import *

import matplotlib.pyplot as plt

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
        diff = torch.fmod(diff + torch.pi / 2, torch.pi) - torch.pi / 2
        loss = torch.mean(torch.amin(torch.mean(diff**2, dim=1) ** (1/2), dim=1))

        return loss 
    

class MSPE(nn.Module):

    def __init__(self, d: int, device:str):

        super().__init__()

        self.d = d
        self.permutation_matrices = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)
        self.permutation_matrices = self.permutation_matrices.to(device)

    def forward(self, angle_pred: torch.Tensor, angle_true: torch.Tensor):

        angle_true_perms = torch.einsum('pmn,bn->bmp', self.permutation_matrices, angle_true)
        diff = angle_true_perms - angle_pred.unsqueeze(-1)
        diff = torch.fmod(diff + torch.pi / 2, torch.pi) - torch.pi / 2
        loss = torch.mean(torch.amin(torch.mean(diff**2, dim=1), dim=1))

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
        loss = torch.mean(torch.amin(torch.mean(diff**2, dim=1), dim=1))
        return loss
    


# class RMSPE_v2(nn.Module):

#     def __init__(self, d: int, device: str, alpha: float = 5.0):

#         super().__init__()

#         self.d = d
#         self.alpha = alpha
#         self.permutation_matrices = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)
#         self.permutation_matrices = self.permutation_matrices.to(device)

#     def forward(self, angle_pred: torch.Tensor, angle_true: torch.Tensor):

#         angle_true_perms = torch.einsum('pmn,bn->bmp', self.permutation_matrices, angle_true)
#         diff = angle_true_perms - angle_pred.unsqueeze(-1)
#         periodic_loss = 1 - torch.exp(- self.alpha * torch.sin(diff) ** 2)
#         periodic_loss = torch.mean(torch.amin(torch.mean(periodic_loss, dim=1), dim=1))
#         return periodic_loss



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
                  gap: float = 0.1, angle_min: float = -torch.pi/2+0.1, angle_max: float = torch.pi/2-0.1):
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



# def MUSIC(X: torch.Tensor, d: int, array: ArrayModelAbstractClass, nbRefines: int=0):

#     m, t = X.shape
#     cov = X @ X.T.conj() / t
#     vals, vecs = torch.linalg.eigh(cov)
#     vecs = vecs[:, vals.sort()[1]]
#     En = vecs [:, :(m-d)]
#     spectrum = 1 / torch.norm(En.conj().transpose(-2, -1) @ array.array_manifold, dim=-2) ** 2
#     peaks_idx = find_peaks_customized(spectrum)
#     peaks_idx = peaks_idx[list(torch.argsort(spectrum[peaks_idx])[-d:])]
#     theta = array.angles_spectrum[peaks_idx]

#     if nbRefines > 0:

#         for _ in range(nbRefines):
    
#             refined_radius = 0.05
#             refined_nbSamples = 100
#             refined_angles_spectrum = [torch.linspace(x-refined_radius, x+refined_radius, refined_nbSamples) for x in theta]
#             refined_spectrums = [1 / torch.norm(En.conj().transpose(-2, -1) @ array.get_steering_vector(angle_spectrum), dim=-2) ** 2 for angle_spectrum in refined_angles_spectrum]
#             refined_peaks_idx = [find_peaks_customized(spectrumi) for spectrumi in refined_spectrums]
#             refined_peaks_idx = [idxi[torch.argmax(spectrumi[idxi])] for idxi, spectrumi in zip(refined_peaks_idx, refined_spectrums)]
#             thetas = [refined_angles_spectrumi[idxi] for idxi, refined_angles_spectrumi in zip(refined_peaks_idx, refined_angles_spectrum)]
#             theta = torch.stack(thetas, dim=0)

#     if theta.shape[0] < d:

#         theta = torch.cat((theta, torch.pi * (torch.rand(d - len(theta.shape[0])) - 1/2)))
#         return theta, spectrum, False
    
#     return theta, spectrum, True



class MUSIC:

    def __init__(self, d: int, array: ArrayModelAbstractClass, 
                 angle_min: float, angle_max: float, nbSamples_spectrum: int):
        
        self.d = d 
        self.angles_spectrum = torch.arange(0, nbSamples_spectrum, 1) * (angle_max - angle_min) / nbSamples_spectrum + angle_min
        self.array_manifold = array.get_steering_vector(self.angles_spectrum)

    def estimate(self, X: torch.Tensor):

        cov = X @ X.T.conj() / X.shape[1]
        vals, vecs = torch.linalg.eigh(cov)
        vecs = vecs[:, vals.sort()[1]]
        En = vecs [:, :-self.d]

        spectrum = 1 / torch.norm(En.conj().transpose(-2, -1) @ self.array_manifold, dim=-2) ** 2
        idx = find_peaks_customized(spectrum)
        idx = idx[list(torch.argsort(spectrum[idx])[-self.d:])]
        theta = self.angles_spectrum[idx]

        if len(idx) < self.d:

            theta = torch.cat((theta, torch.pi * (torch.rand(self.d - len(idx)) - 1/2)))
            return theta, spectrum, False
        
        return theta, spectrum, True
    


class Root_MUSIC:

    def __init__(self, d: int, array: ArrayModelAbstractClass):
    
        self.d = d
        self.m = array.m
        self.lamda = array.lamda
        self.distance = sqrt((array.x[1] - array.x[0]) ** 2 + (array.y[1] - array.y[0]) ** 2)


    def estimate(self, X):
        """
        X: Tensor of shape (B, T, M) where B is batchSize, T is length of signals and M is number of sensors
        """

        Rx = X.transpose(1, 2) @ X.conj() / X.shape[1]

        vals, vecs = torch.linalg.eigh(Rx)
        idx = torch.sort(vals, dim=1)[1].unsqueeze(dim=1).repeat(repeats=(1, self.m, 1))
        vecs = torch.gather(vecs, dim=2, index=idx)
        En = vecs[:, :, :-self.d]

        R = En @ En.conj().transpose(-2, -1)
        coef = torch.zeros(Rx.shape[0], 2 * self.m - 1, dtype=torch.complex64, device=Rx.device)
        for i in range(2 * self.m - 1):
            coef[:, i] = torch.diagonal(R, offset=i - self.m + 1, dim1=-2, dim2=-1).sum(dim=-1)
        coef = coef / coef[:, -1].unsqueeze(1)
        C = torch.zeros(Rx.shape[0], 2 * self.m - 2, 2 * self.m - 2, dtype=torch.complex64, device=Rx.device)
        C[:, :, -1] = - coef[:, :-1]
        C[:, 1:, :-1] = torch.eye(2 * self.m - 3, dtype=torch.complex64, device=Rx.device)
        roots = torch.linalg.eigvals(C)

        roots = roots[torch.abs(roots) < 1]
        roots = roots.reshape(-1, self.m - 1)

        indx_sorted = torch.argsort(torch.abs(torch.abs(roots) - 1), dim=1)
        roots = torch.gather(roots, dim=1, index=indx_sorted)[:, :self.d]
        
        return - torch.arcsin(torch.angle(roots) * self.lamda / (2 * torch.pi * self.distance))



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



def CRLB(array, angles, sources_cov, noise_sigma2, n_snapshots):
    """
    References:
        [1] P. Stoica and A. Nehorai, "Performance study of conditional and
        unconditional direction-of-arrival estimation," IEEE Transactions on
        Acoustics, Speech and Signal Processing, vol. 38, no. 10,
        pp. 1783-1795, Oct. 1990.
    """
    A = array.get_steering_vector(angles)
    D = array.get_first_derivative_steering_vector(angles)
    I = torch.eye(array.m, dtype=torch.complex64)
    R = A @ sources_cov @ A.T.conj() + noise_sigma2 * I
    H = D.T.conj() @ (I - A @ torch.linalg.inv(A.T.conj() @ A) @ A.T.conj()) @ D
    CRB = torch.real(H * (sources_cov @ A.T.conj() @ torch.linalg.inv(R) @ A @ sources_cov).T)
    CRB = torch.linalg.inv(CRB) * (noise_sigma2 / n_snapshots / 2)
    
    return (CRB + CRB.T) / 2



# def CRLB(array, angles, sources_cov, noise_sigma, n_snapshots):
#     """
#     References:
#         [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.

#         [2] M. Wang and A. Nehorai, "Coarrays, MUSIC, and the Cramér-Rao Bound,"
#         IEEE Transactions on Signal Processing, vol. 65, no. 4, pp. 933-946,
#         Feb. 2017.

#         [3] C-L. Liu and P. P. Vaidyanathan, "Cramér-Rao bounds for coprime and
#         other sparse arrays, which find more sources than sensors," Digital
#         Signal Processing, vol. 61, pp. 43-61, 2017.
#     """
#     m = array.m
#     sources_var = torch.diag(sources_cov)
#     A = array.get_steering_vector(angles)
#     D = array.get_first_derivative_steering_vector(angles)
#     R = (A * sources_var) @ A.T.conj() + noise_sigma * torch.eye(m)
#     R_inv = torch.linalg.inv(R)
#     DRD = D.T.conj() @ R_inv @ D
#     DRA = D.T.conj() @ R_inv @ A
#     ARD = A.T.conj() @ R_inv @ D
#     ARA = A.T.conj() @ R_inv @ A
#     PP = torch.outer(sources_var, sources_var)
#     FIM_tt = 2.0 * torch.real((DRD.T * ARA + DRA.conj() * ARD) * PP)
#     FIM_pp = torch.real(ARA.conj().T * ARA)
#     R_inv2 = R_inv @ R_inv
#     FIM_ss = torch.real(torch.trace(R_inv2)).reshape(1, 1)
#     FIM_tp = 2.0 * torch.real(DRA.conj() * (sources_var.unsqueeze(-1) * ARA))
#     FIM_ts = 2.0 * torch.real(sources_var * torch.sum(D.conj() * (R_inv2 @ A), axis=0)).unsqueeze(-1)
#     FIM_ps = torch.real(torch.sum(A.conj() * (R_inv2 @ A), axis=0)).unsqueeze(-1)
#     FIM = torch.cat((torch.cat((FIM_tt, FIM_tp, FIM_ts), dim=1), 
#                      torch.cat((FIM_tp.T.conj(), FIM_pp, FIM_ps), dim=1), 
#                      torch.cat((FIM_ts.T.conj(), FIM_ps.T.conj(), FIM_ss), dim=1)), dim=0)
#     CRB = torch.linalg.inv(FIM)[:m, :m] / n_snapshots
    
#     return (CRB + CRB.T) / 2