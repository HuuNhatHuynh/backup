import torch, matplotlib
from math import sqrt, sin, cos
from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import maximum_filter, label, generate_binary_structure
from itertools import permutations

import matplotlib.pyplot as plt



class ArrayModelAbstractClass(ABC):

    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    def plot(self) -> None:

        assert hasattr(self, "x"), "x must be initialized"
        assert hasattr(self, "y"), "y must be initialized"

        coarray_x = [self.x[i] - self.x[j] for i in range(self.x.shape[0]) for j in range(self.x.shape[0])]
        coarray_y = [self.y[i] - self.y[j] for i in range(self.y.shape[0]) for j in range(self.y.shape[0])]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].scatter(self.x, self.y, color="blue")
        axes[0].axis("equal")
        axes[0].axis("off")
        axes[0].set_title("Array")
        axes[0].set_xlabel("Horizontal")
        axes[0].set_ylabel("Vertical")

        axes[1].scatter(coarray_x, coarray_y, color="green")
        axes[1].axis("equal")
        axes[1].axis("off")
        axes[1].set_title("Difference Coarray")
        axes[1].set_xlabel("Horizontal")
        axes[1].set_ylabel("Vertical")

        plt.tight_layout()
        plt.show()

    def get_steering_vector(self, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:

        assert phi.shape == theta.shape, "phi and theta must be the same size"
        assert hasattr(self, "x"), "x must be initialized"
        assert hasattr(self, "y"), "y must be initialized"
        assert hasattr(self, "lamda"), "lamda must be initialized"

        phaseshift = self.x.unsqueeze(1) * torch.sin(theta).unsqueeze(0) * torch.cos(phi).unsqueeze(0) + \
                     self.y.unsqueeze(1) * torch.sin(theta).unsqueeze(0) * torch.sin(phi).unsqueeze(0)
        
        return torch.exp(1j * 2 * torch.pi * phaseshift / self.lamda)
    
    # def build_coupling_matrix(self, coupling_func):

    #     assert hasattr(self, "x"), "x must be initialized"
    #     assert hasattr(self, "y"), "y must be initialized"
    #     assert hasattr(self, "nbSensors"), "nbSensors must be initialized"

    #     C = torch.eye(self.nbSensors)
    #     for i in range(1, self.nbSensors):
    #         for j in range(i + 1, self.nbSensors):
    #             distance = torch.sqrt((self.x[i] - self.x[j]) ** 2 + (self.y[i] - self.y[j]) ** 2)
    #             coupling_value = coupling_func(distance)
    #             C[i, j] = C[j, i] = coupling_value

    #     return C 
    
    def build_array_manifold(self, phi_min: float = 0,
                                   phi_max: float = torch.pi,
                                   nb_phi: int = 360,
                                   theta_min: float = 0,
                                   theta_max: float = torch.pi / 2,
                                   nb_theta: int = 180) -> None:
        
        self.phi_space = torch.arange(0, nb_phi, 1) * (phi_max - phi_min) / nb_phi + phi_min
        self.theta_space = torch.arange(0, nb_theta, 1) * (theta_max - theta_min) / nb_theta + theta_min
        self.array_manifold = self.get_steering_vector(self.phi_space.repeat(nb_theta), self.theta_space.repeat_interleave(nb_phi))
        self.array_manifold = self.array_manifold.unflatten(dim=1, sizes=(nb_theta, nb_phi))



class URA(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, size_horizontal: int, size_vertical: int):

        self.lamda = lamda
        self.size_horizontal = size_horizontal
        self.size_vertical = size_vertical
        self.nbSensors = size_horizontal * size_vertical
        self.x = torch.arange(0, size_horizontal, 1).repeat(size_vertical) * min_distance
        self.y = torch.arange(0, size_vertical, 1).repeat_interleave(size_horizontal) * min_distance



class NestedArray2D(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, levels_horizontal: list[int], levels_vertical: list[int]):
        
        self.lamda = lamda
        self.min_distance = min_distance
        self.nbSensors_horizontal = sum(levels_horizontal)
        self.nbSensors_vertical = sum(levels_vertical)
        self.nbSensors = self.nbSensors_horizontal * self.nbSensors_vertical

        mult = 1
        self.pos_horizontal = []
        for i in range(len(levels_horizontal)):
            self.pos_horizontal += [(k + 1) * mult for k in range(levels_horizontal[i])]
            mult *= levels_horizontal[i] + 1

        mult = 1
        self.pos_vertical = []
        for i in range(len(levels_vertical)):
            self.pos_vertical += [(k + 1) * mult for k in range(levels_vertical[i])]
            mult *= levels_vertical[i] + 1

        self.x = torch.Tensor(self.pos_horizontal).repeat(sum(levels_vertical)) * min_distance
        self.y = torch.Tensor(self.pos_vertical).repeat_interleave(sum(levels_horizontal)) * min_distance



class OpenBoxArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, size_horizontal: float, size_vertical: float):

        self.lamda = lamda
        self.size_horizontal = size_horizontal
        self.size_vertical = size_vertical
        self.nbSensors = size_horizontal + 2 * size_vertical - 2
        
        self.x = torch.cat((torch.arange(0, size_horizontal, 1), 
                            torch.Tensor([0, size_horizontal - 1]).repeat(size_vertical - 1)), dim=0) * min_distance
        self.y = torch.cat((torch.zeros(size_horizontal), 
                            torch.arange(1, size_vertical, 1).repeat_interleave(2)), dim=0) * min_distance
        


# class UCA(ArrayModelAbstractClass):

#     def __init__(self, lamda: float, radius: float, nbSensors: int):
        
#         self.lamda = lamda
#         self.nbSensors = nbSensors
#         self.x = radius * torch.cos((2 * torch.pi) / nbSensors * torch.arange(nbSensors))
#         self.y = radius * torch.sin((2 * torch.pi) / nbSensors * torch.arange(nbSensors))
        


class HexagonalLatticeArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, size_radius: int):
        
        self.lamda = lamda
        self.nbSensors = 1 + 3 * size_radius * (size_radius - 1)
        self.x = torch.zeros(1)
        self.y = torch.zeros(1)

        for i in range(1, size_radius):
            for j in range(6):
                self.x = torch.cat((self.x, (i * cos(j * torch.pi / 3) + torch.arange(i) * cos((j + 2) * torch.pi / 3)) * min_distance), dim=0)
                self.y = torch.cat((self.y, (i * sin(j * torch.pi / 3) + torch.arange(i) * sin((j + 2) * torch.pi / 3)) * min_distance), dim=0)

class OpenHexagonalLatticeArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, size_radius: int):
        
        self.lamda = lamda
        self.x = torch.zeros(1)
        self.y = torch.zeros(1)

        size_radius = size_radius - 1

        for j in range(6):
            self.x = torch.cat((self.x, (size_radius * cos(j * torch.pi / 3) + torch.arange(size_radius) * cos((j + 2) * torch.pi / 3)) * min_distance), dim=0)
            self.y = torch.cat((self.y, (size_radius * sin(j * torch.pi / 3) + torch.arange(size_radius) * sin((j + 2) * torch.pi / 3)) * min_distance), dim=0)

        self.x = self.x[1:]
        self.y = self.y[1:]

        self.nbSensors = self.x.shape[0]


class HalfOpenBoxArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, min_distance: float, size_horizontal: float, size_vertical: float):

        self.lamda = lamda
        self.size_horizontal = size_horizontal
        self.size_vertical = size_vertical
        self.nbSensors = size_horizontal + 2 * size_vertical - 2
        
        self.x = torch.cat((torch.zeros(1),
                            1 + 2 * torch.arange(0, int((size_horizontal - 3) / 2) + 1, 1),
                            torch.Tensor([size_horizontal - 1]), 
                            torch.Tensor([0, size_horizontal - 1]).repeat(size_vertical - 2),
                            torch.zeros(1), 
                            size_horizontal - 1 - 2 * torch.arange(int((size_horizontal - 2) / 2), 0, -1),
                            torch.Tensor([size_horizontal - 1])), dim=0) * min_distance
        self.y = torch.cat((torch.zeros(int((size_horizontal - 3) / 2) + 3), 
                            torch.arange(1, size_vertical - 1, 1).repeat_interleave(2), 
                            torch.ones(int(size_horizontal / 2) + 1) * (size_vertical - 1)), dim=0) * min_distance
        


def MUSIC_peak_finding_2D(spectrum: torch.Tensor, d: int):

    neighborhood = generate_binary_structure(2, 2)
    local_max = (spectrum == maximum_filter(spectrum, footprint=neighborhood))
    labeled_peaks, _ = label(local_max)
    peak_coords = np.argwhere(labeled_peaks)
    peak_coords_flatten = peak_coords[:, 0] * spectrum.shape[1] + peak_coords[:, 1]
    spectrum_flatten = spectrum.flatten()
    sort_idx = torch.argsort(spectrum_flatten[peak_coords_flatten], descending=True)
    selected_coords_sort = peak_coords_flatten[sort_idx][:d]
    peak_coords_theta = selected_coords_sort // spectrum.shape[1]
    peak_coords_phi = selected_coords_sort % spectrum.shape[1]
    return peak_coords_phi, peak_coords_theta 



def plot_MUSIC_spectrum(fig, axe, name, spectrum, phi_space, theta_space, phi_true, theta_true):

    phi_plot, theta_plot = np.meshgrid(phi_space, theta_space)
    spectrum = 10 * torch.log(spectrum / torch.max(spectrum))
    
    contour = axe.contourf(phi_plot, theta_plot, spectrum, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=axe, label='MUSIC Spectrum')

    peak_coords_phi, peak_coords_theta = MUSIC_peak_finding_2D(spectrum, phi_true.shape[-1])
    phi_predicted, theta_predicted = phi_space[peak_coords_phi], theta_space[peak_coords_theta]

    axe.scatter(phi_true, theta_true, color='red', marker='o', label="target")
    axe.scatter(phi_predicted, theta_predicted, color='orange', marker='x', label="peak")

    axe.set_xlabel('Azimuth (rad)')
    axe.set_ylabel('Elevation (rad)')
    axe.legend()
    axe.set_title(name)



def MUSIC(X: torch.Tensor, d: int, array: ArrayModelAbstractClass):

    Rx = X @ X.T.conj() / X.shape[1]
    _, vecs = torch.linalg.eigh(Rx)
    En = vecs[:, :-d]
    spectrum = 1 / torch.norm(torch.einsum('mi,mjk->ijk', En.conj(), array.array_manifold), dim=0) ** 2
    peak_coords_phi, peak_coords_theta = MUSIC_peak_finding_2D(spectrum, d)
    phi_predicted, theta_predicted = array.phi_space[peak_coords_phi], array.theta_space[peak_coords_theta]

    return phi_predicted, theta_predicted



class RMSPE:

    def __init__(self, d: int):

        self.d = d
        self.perm_mat = torch.stack([torch.stack(list(perm), dim=0) for perm in permutations(torch.eye(d))], dim=0)

    def calculate(self, phi_predicted: torch.Tensor, theta_predicted: torch.Tensor, phi_true: torch.Tensor, theta_true: torch.Tensor):
        
        angles_predicted = torch.stack((phi_predicted, theta_predicted), dim=1)
        angles_true = torch.stack((phi_true, theta_true), dim=1).unsqueeze(0)

        angles_predicted_permute = self.perm_mat @ angles_predicted
        diff = torch.fmod(angles_true - angles_predicted_permute + torch.pi / 2, torch.pi) - torch.pi / 2
        
        return torch.amin(torch.sqrt(torch.mean(diff ** 2, dim=[1, 2])), dim=0)
    


def generate_angles(d: int, angle_min: float, angle_max: float, gap: float = 0.1) -> torch.Tensor:

    while True:
        angles = torch.rand(d) * (angle_max - angle_min) + angle_min
        angles = angles.sort()[0]  
        if torch.min(angles[1:] - angles[:-1]) > gap and angles[-1] - angles[0] < angle_max - angle_min - gap:
            break 
    
    return angles