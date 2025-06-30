import torch
from random import random
from abc import ABC, abstractmethod


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
                                                     - self.x.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1))
            
        return factor * self.get_steering_vector(theta)
    
    def build_array_manifold(self, angle_min: torch.Tensor = - torch.pi / 2,
                                   angle_max: torch.Tensor = torch.pi / 2,
                                   nbSamples: int = 360) -> None:
        
        self.nbSamples_spectrum = nbSamples
        self.angles_spectrum = torch.arange(0, nbSamples, 1) * (angle_max - angle_min) / nbSamples + angle_min
        self.array_manifold = self.get_steering_vector(self.angles_spectrum)

    def build_transform_matrices_from_array_manifold(self):

        self.transform_matrices = self.get_transform_matrix(self.array_manifold)

    def generate_random_mc_coef(self, mc_range: int, mag_min: float = 0.01, mag_max: float = 0.06):

        distances = torch.Tensor([torch.sqrt((self.x[i] - self.x[0]) ** 2 + (self.y[i] - self.y[0]) ** 2) for i in range(1, mc_range)])
        alpha = torch.rand(1) * (mag_max - mag_min) + mag_min
        phis = torch.rand(mc_range - 1) * 2 * torch.pi
        random_mc_coef = alpha / distances * torch.exp(1j * phis)
        random_mc_coef = torch.cat((torch.ones(1), random_mc_coef), dim=0)

        return random_mc_coef
    
    @abstractmethod
    def build_array(self, *args):
        pass

    @abstractmethod
    def generate_random_mc(self, mc_range: int):
        pass

    @abstractmethod
    def get_transform_matrix(self, steering_matrix: torch.Tensor):
        pass



class ULA(ArrayModelAbstractClass):

    def build_array(self, distance: float):

        self.x = torch.arange(0, self.m, 1) * distance
        self.y = torch.zeros(self.m)

    def generate_random_mc(self, mc_range: int):

        coef_mc = torch.rand(mc_range, dtype=torch.complex64)
        coef_mc = torch.nn.functional.pad(coef_mc, (0, self.m - mc_range))
        C = build_symmetric_toeplitz(coef_mc).to(torch.complex64)

        return C

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

    def build_array(self, radius: float):
        
        angles = torch.arange(0, self.m, 1) * 2 * torch.pi / self.m
        self.x = radius * torch.cos(angles)
        self.y = radius * torch.sin(angles)   

    def generate_random_mc(self, mc_range: int):
        
        # coef_mc = torch.rand(mc_range, dtype=torch.complex64)
        coef_mc = self.generate_random_mc_coef(mc_range)
        coef_mc = torch.nn.functional.pad(coef_mc, (0, int(self.m / 2) + 1 - mc_range))
        C = build_symmetric_circulant_toeplitz(coef_mc).to(torch.complex64)

        return C

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
    


def build_symmetric_toeplitz(first_row: torch.Tensor):
    
    m = first_row.shape[-1]
    indices = torch.arange(m)
    toeplitz_matrix = first_row[..., torch.abs(indices - indices.view(-1, 1))]
    
    return toeplitz_matrix



def build_symmetric_circulant_toeplitz(half_first_row: torch.Tensor):

    another_half = torch.flip(half_first_row[..., 1:-1], dims=(-1, ))
    first_row = torch.cat((half_first_row, another_half), dim=-1)
    
    return build_symmetric_toeplitz(first_row)