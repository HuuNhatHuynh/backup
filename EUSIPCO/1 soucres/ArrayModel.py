import torch
from random import random
from abc import ABC, abstractmethod


class ArrayModelAbstractClass(ABC):

    def __init__(self, m: int, lamda: float) -> None:
        
        self.m: int = m
        self.lamda: float = lamda
        self.x: torch.Tensor = torch.empty(0)
        self.y: torch.Tensor = torch.empty(0)

    def get_steering_vector(self, theta: torch.Tensor, return_first_derivative=False) -> torch.Tensor:
        
        if len(theta.shape) == 1:
            steering_vector = torch.exp(-1j * 2 * torch.pi / self.lamda * (self.x.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                                         + self.y.unsqueeze(1) * torch.cos(theta).unsqueeze(0)))
            if return_first_derivative:
                factor = 1j * 2 * torch.pi / self.lamda * (self.y.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                         - self.x.unsqueeze(1) * torch.cos(theta).unsqueeze(0))
            
        if len(theta.shape) == 2:
            steering_vector = torch.exp(-1j * 2 * torch.pi / self.lamda * (self.x.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                                         + self.y.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)))
            if return_first_derivative:
                factor = 1j * 2 * torch.pi / self.lamda * (self.y.reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                         - self.x.reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1))
                
        if return_first_derivative:
            return steering_vector, factor * steering_vector
        else:
            return steering_vector
    
    def build_array_manifold(self, angle_min: torch.Tensor = - torch.pi / 2,
                                   angle_max: torch.Tensor = torch.pi / 2,
                                   nbSamples: int = 360) -> None:
        
        self.nbSamples_spectrum = nbSamples
        self.angles_spectrum = torch.arange(0, nbSamples, 1) * (angle_max - angle_min) / nbSamples + angle_min
        self.array_manifold = self.get_steering_vector(self.angles_spectrum)
    
    @abstractmethod
    def build_sensor_positions(self, *args):
        
        pass



class ULA(ArrayModelAbstractClass):

    def build_sensor_positions(self, distance: float):

        self.x = torch.arange(0, self.m, 1) * distance
        self.y = torch.zeros(self.m)



class ULA_with_noisy_position(ArrayModelAbstractClass):

    def build_sensor_positions(self, distance: float, perturbation: float):
        
        self.x = torch.arange(0, self.m, 1) * distance + (torch.rand(self.m) - 1 / 2) * perturbation
        self.y = torch.zeros(self.m)



class UCA(ArrayModelAbstractClass):

    def build_sensor_positions(self, radius: float):
        
        angles = torch.arange(0, self.m, 1) * 2 * torch.pi / self.m
        self.x = radius * torch.cos(angles)
        self.y = radius * torch.sin(angles)