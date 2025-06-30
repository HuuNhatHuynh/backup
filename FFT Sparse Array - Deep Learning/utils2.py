import torch, matplotlib
from math import sqrt, sin, cos
from abc import ABC, abstractmethod
import numpy as np
from scipy.fft import fft
from scipy.ndimage import maximum_filter, label, generate_binary_structure
from itertools import permutations

import matplotlib.pyplot as plt

class ArrayModelAbstractClass(ABC):

    @abstractmethod
    def __init__(self, *args) -> None:
        super().__init__()

    def build_coarray(self) -> None:

        self.pos_coarray_set = set()
        for i in range(self.nbSensors):
            for j in range(self.nbSensors):
                self.pos_coarray_set.add((self.x[i] - self.x[j], self.y[i] - self.y[j]))

        self.pos_coarray_dict = {}
        for pos in self.pos_coarray_set: self.pos_coarray_dict[pos] = []

        for i in range(self.nbSensors):
            for j in range(self.nbSensors):
                pos = (self.x[i] - self.x[j], self.y[i] - self.y[j])
                self.pos_coarray_dict[pos].append((i, j))

    def plot(self) -> None:

        posx, posy = np.array(self.x), np.array(self.y)
        pos_xy = self.min_distance * self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(posx, posy, s=2)
        axes[0].set_title('array sensor positions')
        axes[0].axis('equal')
        axes[0].grid(True)

        posx, posy = np.array(list(self.pos_coarray_set))[:, 0], np.array(list(self.pos_coarray_set))[:, 1]
        pos_xy = self.min_distance * self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]

        axes[1].scatter(posx, posy, s=2)
        axes[1].set_title('coarray sensor positions')
        axes[1].axis('equal')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def get_steering_vector(self, phi, theta):

        posx, posy = np.array(self.x), np.array(self.y)
        pos_xy = self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]

        phaseshift = self.min_distance * (np.array(posx).reshape(-1, 1) * (np.sin(theta) * np.cos(phi)).reshape(1, -1) 
                                        + np.array(posy).reshape(-1, 1) * (np.sin(theta) * np.sin(phi)).reshape(1, -1))
        
        return  np.exp(1j * 2 * np.pi / self.lamda * phaseshift)
    
    def build_array_manifold(self, phi_min: float = 0,
                                   phi_max: float = np.pi,
                                   nb_phi: int = 360,
                                   theta_min: float = 0,
                                   theta_max: float = np.pi / 2,
                                   nb_theta: int = 180):
        
        self.phi_space = np.arange(0, nb_phi, 1) * (phi_max - phi_min) / nb_phi + phi_min
        self.theta_space = np.arange(0, nb_theta, 1) * (theta_max - theta_min) / nb_theta + theta_min        
        self.array_manifold = self.get_steering_vector(np.repeat(self.phi_space, nb_theta), np.tile(self.theta_space, nb_phi))
        self.array_manifold = self.array_manifold.reshape(-1, nb_phi, nb_theta).transpose(0, 2, 1)

    def estimate_doa_fft(self, X, nbSources, fft_vertical_size=127, fft_horizontal_size=127, show_fft=False):

        cov = X @ X.T.conj() / X.shape[1]
        coarray_covariance = {}
        for pos, indices in self.pos_coarray_dict.items():
            indices = np.array(indices)
            values = np.mean(cov[indices[:, 0], indices[:, 1]])
            coarray_covariance[pos] = values

        R = np.zeros((2*fft_vertical_size+1, 2*fft_horizontal_size+1), dtype=np.complex64)
        for pos in self.pos_coarray_dict.keys():
            R[pos[1]+fft_vertical_size, pos[0]+fft_horizontal_size] = coarray_covariance[pos]

        Rfft = np.abs(fft(fft(R, axis=0), axis=1))
        if show_fft: plt.imshow(Rfft)

        neighborhood = maximum_filter(Rfft, size=3, mode='constant', cval=-np.inf)
        peaks_mask = (Rfft == neighborhood)

        for shift in [(-1,0), (1,0), (0,-1), (0,1)]:
            shifted = np.roll(Rfft, shift, axis=(0,1))
            mask = np.ones_like(Rfft, dtype=bool)
            if shift[0] == -1:
                mask[0,:] = False
            elif shift[0] == 1:
                mask[-1,:] = False
            if shift[1] == -1:
                mask[:,0] = False
            elif shift[1] == 1:
                mask[:,-1] = False
            peaks_mask &= (Rfft > shifted) | ~mask  

        peak_indices = np.argwhere(peaks_mask)
        peak_values = Rfft[peaks_mask]
        sorted_indices = np.argsort(-peak_values)
        top_peaks = [(peak_indices[i]) for i in sorted_indices[:nbSources]] 

        def arctan2_custom(y, x):
            angle = np.arctan2(y, x)
            angle = np.where(angle < 0, angle + np.pi * 2, angle)
            angle = np.where(angle > np.pi, 2 * np.pi - angle, angle)
            return angle 
        
        estimated_phi, estimated_theta = [], []

        for j, i in top_peaks:
            if i <= (2 * fft_horizontal_size + 1) * self.min_distance / self.lamda: 
                b1 = - i / (self.min_distance / self.lamda * (2 * fft_horizontal_size + 1))
            else: b1 = (1 - i / (2 * fft_horizontal_size + 1)) * self.lamda / self.min_distance
            if j <= (2 * fft_vertical_size + 1) * self.min_distance / self.lamda: 
                b2 = - j / (self.min_distance / self.lamda * (2 * fft_vertical_size + 1))
            else: b2 = (1 - j / (2 * fft_vertical_size + 1)) * self.lamda / self.min_distance
            x = np.linalg.solve(self.shear.T, np.array([b1, b2]))
            estimated_theta.append(np.arcsin(np.clip(np.sqrt(x[0] ** 2 + x[1] ** 2), -1, 1)))
            estimated_phi.append(arctan2_custom(- x[1], - x[0]))

        return np.array(estimated_phi), np.array(estimated_theta)
    
    def estimate_doa_music(self, X, nbSources):

        Rx = X @ X.T.conj() / X.shape[1]
        _, vecs = np.linalg.eigh(Rx)
        En = vecs[:, :-nbSources]
        spectrum = 1 / np.linalg.norm(np.einsum('mi,mjk->ijk', np.conj(En), self.array_manifold), axis=0) ** 2

        neighborhood = generate_binary_structure(2, 2)
        local_max = (spectrum == maximum_filter(spectrum, footprint=neighborhood))
        labeled_peaks, _ = label(local_max)
        peak_coords = np.argwhere(labeled_peaks)
        peak_coords_flatten = peak_coords[:, 0] * spectrum.shape[1] + peak_coords[:, 1]
        spectrum_flatten = spectrum.flatten()
        sort_idx = np.argsort(- spectrum_flatten[peak_coords_flatten])
        selected_coords_sort = peak_coords_flatten[sort_idx][:nbSources]
        peak_coords_theta = selected_coords_sort // spectrum.shape[1]
        peak_coords_phi = selected_coords_sort % spectrum.shape[1]

        estimated_phi, estimated_theta = self.phi_space[peak_coords_phi], self.theta_space[peak_coords_theta]

        return estimated_phi, estimated_theta


class UniformRectangularArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):

        self.lamda = lamda
        self.min_distance = min_distance
        self.size_horizontal = size_horizontal
        self.size_vertical = size_vertical
        self.nbSensors = size_horizontal * size_vertical
        self.shear = shear

        self.x = np.tile(np.arange(0, size_horizontal, 1), size_vertical).tolist()
        self.y = np.repeat(np.arange(0, size_vertical, 1), size_horizontal).tolist()


class NestedArray2D(ArrayModelAbstractClass):

    def __init__(self, lamda: float, 
                 min_distance: float, 
                 levels_horizontal: list[int], 
                 levels_vertical: list[int], 
                 shear = np.array([[1, 0], [0, 1]])):
        
        self.lamda = lamda
        self.min_distance = min_distance
        self.nbSensors_horizontal = sum(levels_horizontal)
        self.nbSensors_vertical = sum(levels_vertical)
        self.nbSensors = self.nbSensors_horizontal * self.nbSensors_vertical
        self.shear = shear

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

        self.x = np.tile(np.array(self.pos_horizontal), sum(levels_vertical)).tolist()
        self.y = np.repeat(np.array(self.pos_vertical), sum(levels_horizontal)).tolist()


class OpenBoxArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):

        self.lamda = lamda
        self.min_distance = min_distance
        self.nbSensors = size_horizontal + 2 * (size_vertical - 1)
        self.shear = shear

        self.x, self.y = [], []

        corners = [(0, 0), (size_horizontal - 1, 0), (0, size_vertical - 1), (size_horizontal - 1, size_vertical - 1)]
        g1 = [(i, 0) for i in range(1, size_horizontal - 1)]
        g2 = []
        h1 = [(0, i) for i in range(1, size_vertical - 1)]
        h2 = [(size_horizontal - 1, i) for i in range(1, size_vertical - 1)]

        for s in [corners, g1, g2, h1, h2]:
            for i, j in s:
                self.x.append(i)
                self.y.append(j)



class HalfOpenBoxArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):
        
        self.lamda = lamda
        self.min_distance = min_distance
        self.nbSensors = size_horizontal + 2 * (size_vertical - 1)
        self.shear = shear

        self.x, self.y = [], []

        corners = [(0, 0), (size_horizontal - 1, 0), (0, size_vertical - 1), (size_horizontal - 1, size_vertical - 1)]
        g1 = [(i, 0) for i in range(1, size_horizontal - 1, 2)]
        g2 = [(size_horizontal - 1 - i, size_vertical - 1) for i in range(2, size_horizontal - 1, 2)]
        h1 = [(0, i) for i in range(1, size_vertical - 1)]
        h2 = [(size_horizontal - 1, i) for i in range(1, size_vertical - 1)]

        for s in [corners, g1, g2, h1, h2]:
            for i, j in s:
                self.x.append(i)
                self.y.append(j)


class HalfOpenBoxArray2(ArrayModelAbstractClass):

    def __init__(self, lamda: float, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):
        
        self.lamda = lamda
        self.min_distance = min_distance
        self.nbSensors = size_horizontal + 2 * (size_vertical - 1)
        self.shear = shear

        self.x, self.y = [], []

        corners = [(0, 0), (size_horizontal - 1, 0), (0, size_vertical - 1), (size_horizontal - 1, size_vertical - 1)]
        g1 = [(i, 0) for i in range(1, size_horizontal - 1, 2)]
        g2 = [(size_horizontal - 1 - i, size_vertical - 1) for i in range(2, size_horizontal - 1, 2)]
        h11 = [(0, 1 + 2 * l) for l in range(int((size_vertical - 3) / 2) + 1)] + [(0, size_vertical - 2)]
        h12 = [(1, 2 * l) for l in range(1, int((size_vertical - 3) / 2) + 1)]
        h21 = [(size_horizontal - 1, size_vertical - 1 - j) for _, j in h11]
        h22 = [(size_horizontal - 2, size_vertical - 1 - j) for _, j in h12]

        for s in [corners, g1, g2, h11, h12, h21, h22]:
            for i, j in s:
                self.x.append(i)
                self.y.append(j)



# class HourglassArray(ArrayModelAbstractClass):

#     def __init__(self, lamda: float, 
#                  min_distance: float, 
#                  size_horizontal: int, 
#                  size_vertical: int, 
#                  shear = np.array([[1, 0], [0, 1]])):
        
#         self.lamda = lamda
#         self.min_distance = min_distance
#         self.nbSensors = size_horizontal + 2 * (size_vertical - 1)
#         self.shear = shear

#         self.x, self.y = [], []

#         corners = [(0, 0), (size_horizontal - 1, 0), (0, size_vertical - 1), (size_horizontal - 1, size_vertical - 1)]
#         g1 = [(i, 0) for i in range(1, size_horizontal - 1, 2)]
#         g2 = [(size_horizontal - 1 - i, size_vertical - 1) for i in range(2, size_horizontal - 1, 2)]

#         L = int((size_vertical - 1) / 4) if size_vertical % 2 == 0 else int(size_vertical / 4) + 1
        
#         H1 = []
#         for l in range(1, L + 1):
#             if l == 1:
#                 h1 = []
#                 for p in range(1, int((size_vertical - 1) / 4) + 1):
#                     h1 += [(l - 1, 2 * p), (l - 1, size_vertical - 1 - 2 * p)]
#                 h1 += [(l - 1, 1), (l - 1, size_vertical - 2)]
#                 H1.append(h1)
#             elif size_vertical % 2 == 1:
#                 H1.append([(l - 1, 2 * l - 1), (l - 1, size_vertical - 2 * l)])
#             else:
#                 H1.append([(l - 1, 2 * l - 1), (l - 1, 2 * int(size_vertical / 4) - 2 * l + 3), (l - 1, 2 * int(size_vertical / 4) + 2 * l - 2), (l - 1, size_vertical - 2 * l)])
        
#         H2 = []
#         for l in range(1, L + 1):
#             h2 = []
#             for i, j in H1[l - 1]: h2.append((size_horizontal - 1 - i, j))
#             H2.append(h2)
                
#         for s in [corners, g1, g2] + H1 + H2:
#             for i, j in s:
#                 self.x.append(i)
#                 self.y.append(j)



class HexagonalArray(ArrayModelAbstractClass):

    def __init__(self, lamda: float, 
                 min_distance: float, 
                 shear = np.array([[1, 1/2], [0, sqrt(3)/2]])):

        self.lamda = lamda 
        self.min_distance = min_distance
        self.nbSensors = 102
        self.shear = shear

        self.x, self.y = [], []
        for m in range(-32, 33):
            for n in range(-32, 33):
                if m * m + n * n + m * n in [156, 157, 163, 169, 171, 172, 175, 181]:
                    self.x.append(m)
                    self.y.append(n)


def generate_angles(d: int, angle_min: float, angle_max: float, gap: float = 0.1):

    while True:
        angles = np.random.rand(d) * (angle_max - angle_min) + angle_min
        angles = np.array(sorted(angles))
        if np.min(angles[1:] - angles[:-1]) > gap and angles[-1] - angles[0] < angle_max - angle_min - gap:
            break
    
    return angles


class RMSPE:

    def __init__(self, d: int):

        self.d = d
        self.perm_mat = np.stack([np.stack(list(perm), axis=0) for perm in permutations(np.eye(d))], axis=0)

    def calculate(self, phi_predicted, theta_predicted, phi_true, theta_true):
        
        angles_predicted = np.stack((phi_predicted, theta_predicted), axis=1)
        angles_true = np.expand_dims(np.stack((phi_true, theta_true), axis=1), axis=0)

        angles_predicted_permute = self.perm_mat @ angles_predicted
        diff = np.fmod(angles_true - angles_predicted_permute + np.pi / 2, np.pi) - np.pi / 2
        
        return np.amin(np.sqrt(np.mean(diff ** 2, axis=[1, 2])), axis=0)
    

class RMSPE:

    def __init__(self, d: int):

        self.d = d
        self.perm_mat = np.stack([np.stack(list(perm), axis=0) for perm in permutations(np.eye(d))], axis=0)

    def calculate(self, phi_predicted, theta_predicted, phi_true, theta_true):
        
        angles_predicted = np.stack((phi_predicted, theta_predicted), axis=1)
        angles_true = np.expand_dims(np.stack((phi_true, theta_true), axis=1), axis=0)

        angles_predicted_permute = self.perm_mat @ angles_predicted
        diff = np.fmod(angles_true - angles_predicted_permute + np.pi / 2, np.pi) - np.pi / 2
        
        return np.amin(np.sqrt(np.mean(diff ** 2, axis=(1, 2))), axis=0)