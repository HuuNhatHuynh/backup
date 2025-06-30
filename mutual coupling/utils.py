import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import find_peaks
from math import sqrt, factorial
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
        
        

def generate_direction(d: int, gap: float = 0.1) -> torch.Tensor:
    """
    Generates `D` random directions, ensuring a minimum angular separation.

    Parameters:
    D (int): Number of source directions to generate.
    gap (float): Minimum angular separation between consecutive directions (default is 0.1 radians).

    Returns:
    torch.Tensor: A sorted tensor of size (D) representing the generated directions in radians.
    """

    while True:
        theta = torch.rand(d) * torch.pi - torch.pi / 2
        theta = theta.sort()[0]  
        if torch.min(theta[1:] - theta[:-1]) > gap and theta[-1] - theta[0] < torch.pi - gap:
            break 
    
    return theta



def generate_signal(t: int, d: int, snr: float, array, mc_range: int, coherent: bool):
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

    sigma_noise = 0.1

    theta = generate_direction(d)

    A = array.get_steering_vector(theta)
    
    if not coherent:
        x = (torch.randn(d, t) + 1j * torch.randn(d, t)) / sqrt(2) * 10 ** (snr / 20) * sigma_noise
    else:
        x = (torch.randn(1, t) + 1j * torch.randn(1, t)) / sqrt(2) * 10 ** (snr / 20) * sigma_noise
        x = torch.Tensor.repeat(x, (d, 1))

    n = (torch.randn(array.m, t) + 1j * torch.randn(array.m, t)) / sqrt(2) * sigma_noise

    C = array.generate_random_mc(mc_range)

    return C @ A @ x + n, theta



def generate_data(n: int, t: int, d: int, snr_min: float, snr_max: float, array, mc_range: int, coherent: bool):
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
        y, theta = generate_signal(t, d, snr, array, mc_range, coherent)
        observations.append(y.T) 
        angles.append(theta) 
    
    observations = torch.stack(observations, dim=0)
    angles = torch.stack(angles, dim=0)
    
    return observations, angles



def get_spectrum(En: torch.Tensor, array: ArrayModelAbstractClass) -> torch.Tensor:
    """
    Calculate the MUSIC pseudospectrum based on the given noise subspace and array model.

    Parameters:
    En : torch.Tensor
        The noise subspace, which can have dimensions (M, M-D) or (N, M, M-D). 
        Here, M is the number of sensors, D is the number of signals, and N is the number of samples.

    array : ArrayModel
        An instance of the ArrayModel that defines the array configuration and properties.

    Returns:
    torch.Tensor
        The computed pseudospectrum. It has dimensions (nbSamples_spectrum) 
        for a single sample or (N, nbSamples_spectrum) if multiple samples are processed.
    """
    array.x = array.x.to(En.device)
    array.y = array.y.to(En.device)
    A = array.array_manifold.to(En.device)
    spectrum = 1 / torch.norm(En.conj().transpose(-2, -1) @ A, dim=-2) ** 2
    
    return spectrum



def MUSIC(X: torch.Tensor, d: int, array: ArrayModelAbstractClass):

    m, t = X.shape
    cov = X @ X.T.conj() / t
    vals, vecs = torch.linalg.eigh(cov)
    vecs = vecs[:, vals.sort()[1]]
    En = vecs [:, :(m-d)]
    spectrum = get_spectrum(En, array)
    idx, _ = find_peaks(spectrum)
    idx = idx[list(torch.argsort(spectrum[idx])[-d:])]
    theta = torch.cat((array.angles_spectrum[idx], torch.pi * (torch.rand(d - len(idx)) - 1/2)))

    return theta, spectrum



def permutations(arr):
    """
    Recursively generates all possible permutations of a given list.
    
    Args:
    arr (list): A list of elements for which to compute permutations.

    Returns:
    list: A list containing all permutations of the input list, 
          where each permutation is also a list.
    """
    if len(arr) == 1:
        return [arr]
    
    res = []
    
    for i in range(len(arr)):
        res += [[arr[i]] + perm for perm in permutations(arr[:i] + arr[i+1:])]
    
    return res



def RMSPE(pred, true):
    """
    Calculates the Root Mean Squared Periodic Error (RMSPE) between predicted angles 
    and true angles.

    RMSPE is used to measure the error between periodic (cyclical) quantities 
    such as angles, where values wrap around after a certain threshold.

    Args:
    pred (Tensor): Predicted angles of shape (B, D), where B is the batch size and 
                   D is the dimensionality of the angle space.
    true (Tensor): True angles of shape (B, D), matching the shape of the predicted values.

    Returns:
    Tensor: A scalar tensor representing the RMSPE loss.
    """

    perm = torch.zeros(true.shape[0], factorial(true.shape[1]), true.shape[1])
    
    for i in range(true.shape[0]):
        perm[i] = torch.Tensor(permutations(list(true[i])))
    
    perm = perm.to(device=dev)
    
    diff = torch.fmod(perm - pred.unsqueeze(1) + torch.pi / 2, torch.pi) - torch.pi / 2 
    loss = torch.mean(torch.sqrt((diff ** 2).mean(dim=-1)).amin(dim=-1))
    
    return loss



def RARE(X: torch.Tensor, d: int, mc_range: int, array:ArrayModelAbstractClass):

    m, t = X.shape
    R = X @ X.T.conj() / t
    vals, vecs = torch.linalg.eigh(R)
    vecs = vecs[:, vals.sort()[1]]
    En = vecs [:, :(m-d)]
    W = array.transform_matrices[:, :, :mc_range]
    Q = W.conj().transpose(1, 2) @ En @ En.T.conj() @ W
    # spectrum = torch.abs(1 / torch.linalg.det(Q))
    spectrum = 1 / torch.linalg.eigvalsh(Q)[:, 0]
    idx, _ = find_peaks(spectrum)
    idx = idx[list(torch.argsort(spectrum[idx])[-d:])]
    theta = torch.cat((array.angles_spectrum[idx], torch.pi * (torch.rand(d - len(idx)) - 1/2)))

    return theta, spectrum