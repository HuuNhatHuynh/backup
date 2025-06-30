import torch 
from abc import ABC, abstractmethod
from math import sqrt



class ArrayModelAbstractClass(ABC):

    @abstractmethod
    def __init__(self, *args):
        pass

    def get_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        
        if len(theta.shape) == 1:
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (self.x.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                              + self.y.unsqueeze(1) * torch.cos(theta).unsqueeze(0)))
        
        if len(theta.shape) == 2:
            return torch.exp(-1j * 2 * torch.pi / self.lamda * (self.x.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                              + self.y.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)))
    
    
    def build_array_manifold(self, angle_min: torch.Tensor = - torch.pi / 2,
                                   angle_max: torch.Tensor = torch.pi / 2,
                                   nbSamples: int = 360) -> None:
        
        self.nbSamples_spectrum = nbSamples
        self.angles_spectrum = torch.arange(0, nbSamples, 1) * (angle_max - angle_min) / nbSamples + angle_min
        self.array_manifold = self.get_steering_vector(self.angles_spectrum)



class ULA(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, m: int):

        self.lamda = lamda
        self.min_distance = min_distance
        self.x = torch.arange(0, m, 1) * min_distance
        self.y = torch.zeros_like(self.x)
        


class NestedArray1D(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, levels: list[int]):

        self.lamda = lamda
        self.min_distance = min_distance
        self.levels = levels

        mult = 1
        pos = []
        for i in range(len(levels)):
            pos = pos + [(k + 1) * mult for k in range(levels[i])]
            mult = mult * levels[i] + 1

        self.pos = pos 

        self.x = torch.Tensor(pos) * min_distance
        self.y = torch.zeros_like(self.x)
        


def generate_direction(d: int, angle_min: float, angle_max: float, gap: float = 0.1) -> torch.Tensor:
    """
    Generates `D` random directions, ensuring a minimum angular separation.

    Parameters:
    D (int): Number of source directions to generate.
    gap (float): Minimum angular separation between consecutive directions (default is 0.1 radians).

    Returns:
    torch.Tensor: A sorted tensor of size (D) representing the generated directions in radians.
    """

    while True:
        theta = torch.rand(d) * (angle_max - angle_min) + angle_min
        theta = theta.sort()[0]  
        if torch.min(theta[1:] - theta[:-1]) > gap and theta[-1] - theta[0] < torch.pi - gap:
            break 
    
    return theta



def generate_signal(t: int, d: int, snr: float, array, coherent: bool):
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

    theta = generate_direction(d, angle_min=torch.pi/4, angle_max=3*torch.pi/4)

    A = array.get_steering_vector(theta)
    
    if not coherent:
        x = (torch.randn(d, t) + 1j * torch.randn(d, t)) / sqrt(2) * 10 ** (snr / 20) * sigma_noise
    else:
        x = (torch.randn(1, t) + 1j * torch.randn(1, t)) / sqrt(2) * 10 ** (snr / 20) * sigma_noise
        x = torch.Tensor.repeat(x, (d, 1))

    n = (torch.randn(array.m, t) + 1j * torch.randn(array.m, t)) / sqrt(2) * sigma_noise

    return A @ x + n, theta