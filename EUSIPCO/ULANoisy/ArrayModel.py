import torch
from random import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt



class ArrayModelAbstractClass(ABC):

    def __init__(self, m: int, lamda: float) -> None:
        
        self.m: int = m
        self.lamda: float = lamda
        self.x: torch.Tensor = torch.empty(0)
        self.y: torch.Tensor = torch.empty(0)
        self.del_x = torch.zeros(self.m)
        self.del_y = torch.zeros(self.m)

    def pertube(self, del_x: torch.Tensor, del_y: torch.Tensor) -> None:

        self.del_x = del_x
        self.del_y = del_y

    def plot_array(self) -> None:

        plt.scatter(self.x, self.y, color="green", s=10, label="True position", marker="o")
        plt.scatter(self.x + self.del_x, self.y + self.del_y, color="red", s=10, label="Perturbed position", marker='x')
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.title("Array")
        plt.xlabel("Horizontal")
        plt.ylabel("Vertical")
        plt.ylim([-self.lamda, self.lamda])
        plt.tight_layout()
        plt.show()

    def get_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        
        if len(theta.shape) == 1:
            steering_vector = torch.exp(-1j * 2 * torch.pi / self.lamda * ((self.x + self.del_x).unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                                         + (self.y + self.del_y).unsqueeze(1) * torch.cos(theta).unsqueeze(0)))
            
        if len(theta.shape) == 2:
            steering_vector = torch.exp(-1j * 2 * torch.pi / self.lamda * ((self.x + self.del_x).reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                                         + (self.y + self.del_y).reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1)))
        return steering_vector
        
    def get_first_derivative_steering_vector(self, theta: torch.Tensor) -> torch.Tensor:
        
        if len(theta.shape) == 1:
                factor = 1j * 2 * torch.pi / self.lamda * ((self.y + self.del_y).unsqueeze(1) * torch.sin(theta).unsqueeze(0)
                                                         - (self.x + self.del_x).unsqueeze(1) * torch.cos(theta).unsqueeze(0))
            
        if len(theta.shape) == 2:
                factor = 1j * 2 * torch.pi / self.lamda * ((self.y + self.del_y).reshape(1, -1, 1) * torch.sin(theta).unsqueeze(1)
                                                         - (self.x + self.del_x).reshape(1, -1, 1) * torch.cos(theta).unsqueeze(1))
                
        return factor * self.get_steering_vector(theta)
    
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



class NestedArray1D(ArrayModelAbstractClass):
     
    def build_sensor_positions(self, distance: float, levels: list[int]):
        
        assert self.m == sum(levels), "The number of sensors must be equal to the sum of the levels"

        mult = 1
        pos = []
        for i in range(len(levels)):
            pos += [(k + 1) * mult for k in range(levels[i])]
            mult *= (levels[i] + 1)

        self.x = distance * torch.tensor(pos)
        self.y = torch.zeros(self.m)