import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import find_peaks
from math import sqrt, log
from random import random, randint
from abc import ABC, abstractmethod
from itertools import permutations

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

class ArrayModelAbstractClass(ABC):

    def __init__(self, m: int, lamda: float) -> None:
        
        self.m: int = m
        self.lamda: float = lamda
        self.x: torch.Tensor = None
        self.y: torch.Tensor = None

    def get_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        
        if len(theta.shape) == 1:
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (self.x.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                              + self.y.unsqueeze(1) * torch.cos(theta).unsqueeze(0)))
        
        if len(theta.shape) == 2:
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (self.x.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                              + self.y.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)))
        
    def get_first_derivative_steering_vector(self, theta: torch.Tensor):

        if len(theta.shape) == 1:
            factor = 1j * 2 * torch.pi / self.lamda * (self.y.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                     - self.x.unsqueeze(1) * torch.cos(theta).unsqueeze(0))
        
        if len(theta.shape) == 2:
            factor = 1j * 2 * torch.pi / self.lamda * (self.y.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                     + self.x.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1))
            
        return factor * self.get_steering_vector(theta)
    
    def build_array_manifold(self, angle_min: torch.Tensor = - torch.pi / 2,
                                   angle_max: torch.Tensor = torch.pi / 2,
                                   nbSamples: int = 360) -> None:
        
        self.nbSamples_spectrum = nbSamples
        self.angles_spectrum = torch.arange(0, nbSamples, 1) * (angle_max - angle_min) / nbSamples + angle_min
        self.array_manifold = self.get_steering_vector(self.angles_spectrum)

    def build_transform_matrices(self):

        self.transform_matrices = self.get_transform_matrix(self.array_manifold)
    
    @abstractmethod
    def build_sensor_positions(self, *args):
        pass

    @abstractmethod
    def get_transform_matrix(self, steering_matrix: torch.Tensor):
        pass



class ULA(ArrayModelAbstractClass):

    def build_sensor_positions(self, distance: float):

        self.x = torch.arange(0, self.m, 1) * distance
        self.y = torch.zeros(self.m)

    def get_transform_matrix(self, steering_matrix: torch.Tensor):

        transform_matrix = torch.zeros(steering_matrix.shape[-1], self.m, self.m, dtype=torch.complex64)

        for i in range(self.m):
            for j in range(self.m - i):
                transform_matrix[..., i, j] += steering_matrix[i + j]

        for j in range(1, self.m):
            for i in range(j, self.m):
                transform_matrix[..., i, j] += steering_matrix[i - j]

        return transform_matrix



class UCA(ArrayModelAbstractClass):

    def build_sensor_positions(self, radius: float):
        
        angles = torch.arange(0, self.m, 1) * 2 * torch.pi / self.m
        self.x = radius * torch.cos(angles)
        self.y = radius * torch.sin(angles)   

    def get_transform_matrix(self, steering_matrix: torch.Tensor):

        l = int(self.m / 2) + 1
        transform_matrix = torch.zeros(steering_matrix.shape[-1], self.m, l, dtype=torch.complex64)

        for i in range(self.m):
            for j in range(min(self.m - i, l)):
                transform_matrix[..., i, j] += steering_matrix[i + j]

        for j in range(1, l):
            for i in range(j, self.m):
                transform_matrix[..., i, j] += steering_matrix[i - j]
        
        for i in range(l - (self.m%2==0) - 1):
            for j in range(i + 1, l - (self.m%2==0)):
                transform_matrix[..., i, j] += steering_matrix[self.m + i - j]

        for j in range(1, l - (self.m%2==0)):
            for i in range(self.m - j, self.m):
                transform_matrix[..., i, j] += steering_matrix[i + j - self.m]

        return transform_matrix



class RMSPE_fixed_nbSources:

    def __init__(self, d: int, device):

        self.d = d
        self.permutation_matrices = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)
        self.permutation_matrices = self.permutation_matrices.to(device)

    def calculate(self, angle_pred: torch.Tensor, angle_true: torch.Tensor):

        angle_true_perms = torch.einsum('pmn,bn->bmp', self.permutation_matrices, angle_true)
        diff = torch.fmod(angle_true_perms - angle_pred.unsqueeze(-1) + torch.pi / 2, torch.pi) - torch.pi / 2
        loss = torch.mean(torch.amin(torch.mean(diff**2, dim=1) ** (1/2), dim=1))

        return loss 



class RMSPE_varied_nbSources:

    def __init__(self, dmin: int, dmax: int, device):

        self.d = [i for i in range(dmin, dmax+1)]
        self.permutation_matrices = []

        for i in self.d:

            id = torch.eye(i)
            perms = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(id)], dim=0)
            self.permutation_matrices.append(perms.to(device))

    def calculate(self, angle_pred: torch.Tensor, angle_true: torch.Tensor):

        nbSourcesTrue = torch.argmax(angle_true, dim=1)

        loss = 0.0
        
        for i, permutation_matrices in zip(self.d, self.permutation_matrices):

            idx = (nbSourcesTrue == i)
            pred = angle_pred[idx, :i].unsqueeze(-1)
            perm = torch.einsum('pmn,bn->bmp', permutation_matrices, angle_true[idx, :i])
            diff = torch.fmod(perm - pred + torch.pi / 2, torch.pi) - torch.pi / 2 
            loss = loss + torch.sum(torch.amin(torch.mean(diff**2, dim=1) ** (1/2), dim=1))

        loss = loss / angle_true.shape[0]

        return loss
    


class RMSPE_varied_nbSources_test:

    def __init__(self, dmin: int, dmax: int, device):

        self.d = [i for i in range(dmin, dmax+1)]
        self.loss = [[] for _ in range(dmin, dmax+1)]
        self.permutation_matrices = []
        self.device = device

        for i in self.d:

            id = torch.eye(i)
            perms = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(id)], dim=0)
            self.permutation_matrices.append(perms.to(device))

    def calculate(self, angle_pred: torch.Tensor, nbSources_pred_label: torch.Tensor, angle_true: torch.Tensor):

        if nbSources_pred_label is not None:
            
            nbSourcesPred = torch.argmax(nbSources_pred_label, dim=1)

            for i in self.d:

                idx = (nbSourcesPred == i)
                angle_pred[idx, i:] = torch.rand(idx.int().sum(), angle_pred.shape[1] - i).to(self.device) * torch.pi - torch.pi / 2

        nbSourcesTrue = torch.argmax(angle_true, dim=1)
        
        for i, loss, permutation_matrices in zip(self.d, self.loss, self.permutation_matrices):

            idx = (nbSourcesTrue == i)
            pred = angle_pred[idx, :i].unsqueeze(-1)
            perm = torch.einsum('pmn,bn->bmp', permutation_matrices, angle_true[idx, :i])
            diff = torch.fmod(perm - pred + torch.pi / 2, torch.pi) - torch.pi / 2 
            loss.append(torch.mean(torch.amin(torch.mean(diff**2, dim=1) ** (1/2), dim=1)))

    def resume(self):

        for i, d in enumerate(self.d):
            print("for d = {}, RMSPE is {}".format(d, sum(self.loss[i]) / len(self.loss[i])))
        


class DATASET(Dataset):

    def __init__(self, observations: torch.Tensor, angles: torch.Tensor, labels: torch.Tensor):
        
        self.observations = observations
        self.angles = angles
        self.labels = labels

    def __getitem__(self, idx):

        return self.observations[idx], self.angles[idx], self.labels[idx]
    
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

    # return torch.rand(d) * torch.pi - torch.pi / 2
    
    return theta



def generate_signal(t: int, d: int, snr: float, array, coherent: bool, C: torch.Tensor = None):
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

    return C @ A @ x + n, theta



def generate_data(n: int, t: int, dmin: int, dmax: int, SNRmin: float, SNRmax: float, array, coherent: bool, C: torch.Tensor = None):
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
    labels = torch.zeros(n, dmax - dmin + 1)
    
    for i in range(n):

        d = dmin + i % (dmax - dmin + 1)
        snr = random() * (SNRmax - SNRmin) + SNRmin
        y, theta = generate_signal(t, d, snr, array, coherent, C)
        observations.append(y.T) 
        angles.append(F.pad(theta, (0, array.m - 1 - d), mode='constant', value=torch.pi))
        labels[i, d - dmin] = 1.0 
    
    observations = torch.stack(observations, dim=0)
    angles = torch.stack(angles, dim=0)
    
    return observations, angles, labels



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



def AIC(X: torch.Tensor):
    
    _, t, m = X.shape
    cov = X.transpose(1, 2) @ X.conj() / t
    vals = torch.linalg.eigvalsh(cov)
    Ld = [t * (m - d) * (torch.log(torch.mean(vals[:, :(m - d)], dim=1)) 
                       - torch.mean(torch.log(vals[:, :(m - d)]), dim=1)) for d in range(m)]
    reg = [d * (2 * m - d) for d in range(m)]
    Ld = torch.stack(Ld, dim=1) + torch.Tensor(reg).reshape(1, -1)
    nbSources = torch.argmin(Ld, dim=1)

    return nbSources



def MDL(X: torch.Tensor):

    _, t, m = X.shape
    cov = X.transpose(1, 2) @ X.conj() / t
    vals = torch.linalg.eigvalsh(cov)
    Ld = [t * (m - d) * (torch.log(torch.mean(vals[:, :(m - d)], dim=1)) 
                       - torch.mean(torch.log(vals[:, :(m - d)]), dim=1)) for d in range(m)]
    reg = [(d * (2 * m - d) + 1) * log(t) / 2 for d in range(m)]
    Ld = torch.stack(Ld, dim=1) + torch.Tensor(reg).reshape(1, -1)
    nbSources = torch.argmin(Ld, dim=1)

    return nbSources



def build_symmetric_toeplitz(first_row: torch.Tensor):
    
    m = first_row.shape[-1]
    indices = torch.arange(m)
    toeplitz_matrix = first_row[..., torch.abs(indices - indices.view(-1, 1))]
    
    return toeplitz_matrix



def build_symmetric_circulant_toeplitz(half_first_row: torch.Tensor):

    another_half = torch.flip(half_first_row[..., 1:-1], dims=(-1, ))
    first_row = torch.cat((half_first_row, another_half), dim=-1)
    
    return build_symmetric_toeplitz(first_row)