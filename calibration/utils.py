import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import find_peaks
from math import sqrt, factorial

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nbSamples_spectrum = 360
angles_spectrum = torch.arange(0, nbSamples_spectrum, 1) * torch.pi / nbSamples_spectrum - torch.pi / 2



class DATASET(Dataset):

    def __init__(self, observations: torch.Tensor, angles: torch.Tensor, perturbation_x: torch.Tensor, perturbation_y: torch.Tensor):
        
        self.observations = observations
        self.angles = angles
        self.perturbation_x = perturbation_x
        self.perturbation_y = perturbation_y

    def __getitem__(self, idx):

        return self.observations[idx], self.angles[idx], self.perturbation_x[idx], self.perturbation_y[idx]
    
    def __len__(self):

        return len(self.observations)



class ArrayModel:

    def __init__(self, m: int, lamda: float):
        """
        Initializes the array model with sensor configuration.

        Parameters:
        M (int): Number of sensors in the array.
        lamda (float): Wavelength of the incident signal.
        """
        
        self.m = m
        self.lamda = lamda
        self.array_x = torch.arange(0, m, 1) * lamda / 2  
        self.array_y = torch.zeros(m)

        self.array_x = torch.arange(0, self.m, 1) * self.lamda / 2  
        self.array_y = torch.zeros(self.m)  

    def update_position(self, del_x: torch.Tensor, del_y: torch.Tensor):
        """
        Updates the assumed sensor positions in the array based on the provided displacements.

        Parameters:
        del_x (torch.Tensor): Displacement to be added to the x-coordinates of the sensors.
        del_y (torch.Tensor): Displacement to be added to the y-coordinates of the sensors.
        """
        
        self.array_x = self.array_x + del_x
        self.array_y = self.array_y + del_y
    
    def get_steering_vector(self, theta: torch.Tensor, del_x: torch.Tensor, del_y: torch.Tensor) -> torch.Tensor:
        """
        Computes the steering vector for the given incident angle(s).

        Parameters:
        theta (torch.Tensor): Incident angle(s) of the incoming signal. Can be of size (D) for a single batch 
                              or (B, D) for multiple batches.

        Returns:
        torch.Tensor: Steering vector of size (M, D) for a single batch or (B, M, D) for multiple batches.
        """
        
        x = self.array_x + del_x
        y = self.array_y + del_y
        
        if (len(theta.shape) == 1) and (len(x.shape) == 1):
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (x.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                               + y.unsqueeze(1) * torch.cos(theta).unsqueeze(0)))
        
        if (len(theta.shape) == 1) and (len(x.shape) == 2):
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (x.unsqueeze(2) * torch.sin(theta).reshape(1, 1, -1)
                                                               + y.unsqueeze(2) * torch.cos(theta).reshape(1, 1, -1)))
        
        if (len(theta.shape) == 2) and (len(x.shape) == 1):
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (x.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                               + y.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)))
        
        if (len(theta.shape) == 2) and (len(x.shape) == 2):
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (x.unsqueeze(2) * torch.sin(theta).unsqueeze(1)
                                                               + y.unsqueeze(2) * torch.cos(theta).unsqueeze(1)))
        
        


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



def generate_signal(t: int, d: int, snr: float, array: ArrayModel, coherent: bool):
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

    sigma_noise = 0.01
    perturbation_coeff = 0.2

    theta = generate_direction(d)

    del_x = (2 * torch.rand(array.m) - 1) * perturbation_coeff * array.lamda / 2
    del_y = (2 * torch.rand(array.m) - 1) * perturbation_coeff * array.lamda / 2
    
    # del_x = torch.randn(array.m) * perturbation_coeff * array.lamda / 2
    # del_y = torch.randn(array.m) * perturbation_coeff * array.lamda / 2

    A = array.get_steering_vector(theta, del_x, del_y)
    
    if not coherent:
        x = (torch.randn(d, t) + 1j * torch.randn(d, t)) / sqrt(2) * 10 ** (snr / 20) * sigma_noise
    else:
        x = (torch.randn(1, t) + 1j * torch.randn(1, t)) / sqrt(2) * 10 ** (snr / 20) * sigma_noise
        x = torch.Tensor.repeat(x, (d, 1))

    n = (torch.randn(array.m, t) + 1j * torch.randn(array.m, t)) / sqrt(2) * sigma_noise

    return A @ x + n, theta, del_x, del_y



def generate_data(n: int, t: int, d: int, snr: float, array: ArrayModel, coherent: bool = False):
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
        - `perturbations_x`: Tensor of size (N, M) representing the perturbations in the x-coordinates of the array elements.
        - `perturbations_y`: Tensor of size (N, M) representing the perturbations in the y-coordinates of the array elements.
    """
    
    observations = []
    angles = []
    perturbations_x = []
    perturbations_y = []
    
    for _ in range(n):
        y, theta, del_x, del_y = generate_signal(t, d, snr, array, coherent=coherent)
        observations.append(y.T) 
        angles.append(theta) 
        perturbations_x.append(del_x) 
        perturbations_y.append(del_y)
    
    observations = torch.stack(observations, dim=0)
    angles = torch.stack(angles, dim=0)
    perturbations_x = torch.stack(perturbations_x, dim=0)
    perturbations_y = torch.stack(perturbations_y, dim=0)
    
    return observations, angles, perturbations_x, perturbations_y



def get_spectrum(En: torch.Tensor, array: ArrayModel, del_x: torch.Tensor, del_y: torch.Tensor) -> torch.Tensor:
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
    array.array_x = array.array_x.to(En.device)
    array.array_y = array.array_y.to(En.device)
    A = array.get_steering_vector(angles_spectrum.to(En.device), del_x, del_y)
    spectrum = 1 / torch.norm(En.conj().transpose(-2, -1) @ A, dim=1) ** 2
    
    return spectrum



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



def NMSE(x_pred: torch.Tensor, x_true: torch.Tensor):

    mse = torch.mean((x_pred - x_true) ** 2)
    variance = torch.mean((x_true - torch.mean(x_true)) ** 2)
    return mse / variance