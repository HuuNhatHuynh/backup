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