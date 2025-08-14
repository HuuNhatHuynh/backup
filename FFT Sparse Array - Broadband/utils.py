import matplotlib
from math import sqrt, sin, cos
from abc import ABC, abstractmethod
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.ndimage import maximum_filter, label, generate_binary_structure
from itertools import permutations

import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
        for pos in self.pos_coarray_set: 
            self.pos_coarray_dict[pos] = []

        for i in range(self.nbSensors):
            for j in range(self.nbSensors):
                pos = (self.x[i] - self.x[j], self.y[i] - self.y[j])
                self.pos_coarray_dict[pos].append((i, j))

    def plot(self, savename=None) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(posx, posy, s=2)
        axes[0].set_title('array sensor positions')
        axes[0].axis('equal')
        axes[0].axis('off')
        axes[0].grid(True)

        posx, posy = np.array(list(self.pos_coarray_set))[:, 0], np.array(list(self.pos_coarray_set))[:, 1]
        pos_xy = self.min_distance * self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]

        axes[1].scatter(posx, posy, s=2)
        axes[1].set_title('coarray sensor positions')
        axes[1].axis('equal')
        axes[1].axis('off')
        axes[1].grid(True)

        plt.tight_layout()
        if savename: plt.savefig(savename+".pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

    def plot_coarray_weight(self, plot3D=False) -> None:

        posz = np.array([len(self.pos_coarray_dict[pos]) for pos in self.pos_coarray_set])

        if plot3D:

            points = go.Scatter3d(x=self.posx, y=self.posy, z=posz,
                                mode='markers',
                                marker=dict(size=4, color=posz, colorscale='viridis'),
                                name='co-array weight')
            
            lines = []
            for xi, yi, zi in zip(self.posx, self.posy, posz):
                lines.append(go.Scatter3d(
                    x=[xi, xi],
                    y=[yi, yi],
                    z=[0, zi],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    opacity=0.3,
                    showlegend=False))
                
            fig = go.Figure(data=[points] + lines)
            fig.update_layout(
                title='3D difference co-array weight plot',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='weight'))

            fig.update_layout(width=800, height=800)

        else:

            fig = go.Figure(data=go.Scatter(x=self.posx, y=self.posy,
                            mode='markers',
                            marker=dict(
                            size=6,
                            color=posz,
                            colorscale='plasma',
                            colorbar=dict(title='z'),
                            showscale=True)))
            
            fig.update_layout(title='2D difference co-array weight plot',
                              xaxis_title='x', yaxis_title='y',
                              width=800, height=800)

        fig.show()

    def get_steering_vector(self, freq, phi, theta):

        posx, posy = np.array(self.x), np.array(self.y)
        pos_xy = self.min_distance * self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]

        phaseshift = np.array(posx).reshape(1, -1, 1) * (np.sin(theta) * np.cos(phi)).reshape(1, 1, -1) \
                   + np.array(posy).reshape(1, -1, 1) * (np.sin(theta) * np.sin(phi)).reshape(1, 1, -1)
        
        return  np.exp(- 1j * 2 * np.pi * freq.reshape(-1, 1, 1) * phaseshift / 3e8)
    
    def build_array_manifold(self, phi_min: float = 0,
                                   phi_max: float = np.pi,
                                   nb_phi: int = 720,
                                   theta_min: float = 0,
                                   theta_max: float = np.pi / 2,
                                   nb_theta: int = 360):
        
        self.phi_space = np.arange(0, nb_phi, 1) * (phi_max - phi_min) / nb_phi + phi_min
        self.theta_space = np.arange(0, nb_theta, 1) * (theta_max - theta_min) / nb_theta + theta_min        
        self.array_manifold = lambda freq: self.get_steering_vector(np.array([freq]), 
                                                                    np.repeat(self.phi_space, nb_theta), 
                                                                    np.tile(self.theta_space, nb_phi)).reshape(self.nbSensors, nb_phi, nb_theta)

    def generate_broadband_data(self, sources, phi, theta, snr, nbSnapshots, fSampling):
    
        n = (np.random.randn(self.nbSensors, nbSnapshots) + 1j * np.random.randn(self.nbSensors, nbSnapshots)) / sqrt(2) * 10 ** (-snr/20)
        freqs = fftfreq(sources.shape[1], 1/fSampling)
        A = self.get_steering_vector(freqs, phi, theta)
        S = fft(sources, axis=1)
        X = np.einsum('nmd,dn->mn', A, S)
        x = ifft(X, axis=1)[:, :nbSnapshots] + n
        
        return x
    
    def get_covariances(self, x, nbChunks, fmin, fmax, fsampling, fft_size):

        freqs = fftfreq(fft_size, 1/fsampling)
        idx = np.where((freqs >= fmin) & (freqs <= fmax))
        x_temporal_fft = fft(x.reshape(self.nbSensors, nbChunks, -1), n=fft_size, axis=2).transpose(2, 0, 1)
        x_temporal_fft = x_temporal_fft[idx]
        freqs = freqs[idx]
        Rx = x_temporal_fft @ x_temporal_fft.conj().transpose(0, 2, 1) / nbChunks

        return Rx, freqs
    
    def estimate_doa_spatial_fft_autofocusing(self, Rx, freqs, nbSources, fft_vertical_size, fft_horizontal_size):

        U, _, Vh = np.linalg.svd(Rx)
        Uref = U[U.shape[0]//2]
        ref_freq = freqs[U.shape[0]//2]
        T = Uref @ U.conj().transpose(0, 2, 1) / sqrt(len(freqs))
        Rx_autofocus = np.sum(T @ Rx @ T.conj().transpose(0, 2, 1), axis=0)

        coarray_covariance = {}
        for pos, indices in self.pos_coarray_dict.items():
            indices = np.array(indices)
            values = np.mean(Rx_autofocus[indices[:, 0], indices[:, 1]])
            coarray_covariance[pos] = values

        R_padded = np.zeros((2*fft_vertical_size+1, 2*fft_horizontal_size+1), dtype=np.complex64)
        for pos in self.pos_coarray_dict.keys():
            R_padded[pos[1]+fft_vertical_size, pos[0]+fft_horizontal_size] = coarray_covariance[pos]

        R_spatial_fft = np.abs(fft(fft(R_padded, axis=0), axis=1))

        top_peaks_init = get_top_peaks(R_spatial_fft, nbSources)
        I = 3e8 / (self.min_distance * ref_freq) * ((top_peaks_init[:, 1] >= fft_horizontal_size + 1) 
                                                  - (top_peaks_init[:, 1] / (2 * fft_horizontal_size + 1)))
        J = 3e8 / (self.min_distance * ref_freq) * ((top_peaks_init[:, 0] >= fft_horizontal_size + 1) 
                                                  - (top_peaks_init[:, 0] / (2 * fft_horizontal_size + 1)))
        sol = np.linalg.solve(self.shear.T, np.stack((I, J), axis=0))
        last_estimations_u = sol[0]
        last_estimations_v = sol[1]

        estimated_theta = np.arcsin(np.clip(np.sqrt(last_estimations_u ** 2 + last_estimations_v ** 2), -1, 1))
        estimated_phi = np.arctan2(last_estimations_v, last_estimations_u)
        estimated_phi = np.where(estimated_phi < 0, estimated_phi + np.pi * 2, estimated_phi)
        estimated_phi = np.where(estimated_phi > np.pi, 2 * np.pi - estimated_phi, estimated_phi)

        return estimated_phi, estimated_theta
    
    def estimate_doa_music_autofocusing(self, Rx, freqs, nbSources):

        U, _, Vh = np.linalg.svd(Rx)
        Uref = U[U.shape[0]//2]
        ref_freq = freqs[U.shape[0]//2]
        T = Uref @ U.conj().transpose(0, 2, 1) / sqrt(len(freqs))
        Rx_autofocus = np.sum(T @ Rx @ T.conj().transpose(0, 2, 1), axis=0)

        _, vecs = np.linalg.eigh(Rx_autofocus)
        En = vecs[:, :-nbSources]
        array_manifold = self.array_manifold(ref_freq)
        spectrum = 1 / np.linalg.norm(np.einsum('mi,mjk->ijk', np.conj(En), array_manifold), axis=0) ** 2

        top_peaks_init = get_top_peaks(spectrum, nbSources)
        estimated_phi, estimated_theta = self.phi_space[top_peaks_init[:, 0]], self.theta_space[top_peaks_init[:, 1]]

        return estimated_phi, estimated_theta
    
    def get_focusing_rcsm_init(self, freqs, ref_freq):

        posx, posy = np.array(self.x), np.array(self.y)
        pos_xy = self.min_distance * self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]
        
        Q = 4 * np.sinc(2 / 3e8 * (ref_freq * posx.reshape(1, 1, -1) - freqs.reshape(-1, 1, 1) * posx.reshape(1, -1, 1))) \
              * np.sinc(2 / 3e8 * (ref_freq * posy.reshape(1, 1, -1) - freqs.reshape(-1, 1, 1) * posy.reshape(1, -1, 1))) 
        
        U, _, Vh = np.linalg.svd(Q)
        
        T = Vh.conj().transpose(0, 2, 1) @ U.conj().transpose(0, 2, 1)
        
        return T
    
    def get_focusing_rcsm_iter(self, freqs, ref_freq, last_estimations_u, last_estimations_v, iteration, p):

        d = last_estimations_u.shape[0]

        posx, posy = np.array(self.x), np.array(self.y)
        pos_xy = self.min_distance * self.shear @ np.stack((posx, posy), axis=0)
        posx, posy = pos_xy[0], pos_xy[1]

        A = np.exp(- 1j * 2 * np.pi * freqs.reshape(-1, 1, 1) / 3e8 * (np.array(posx).reshape(1, -1, 1) * last_estimations_u.reshape(1, 1, -1) 
                                                                     + np.array(posy).reshape(1, -1, 1) * last_estimations_v.reshape(1, 1, -1)))
        
        Aref = np.exp(- 1j * 2 * np.pi * ref_freq / 3e8 * (np.array(posx).reshape(-1, 1) * last_estimations_u.reshape(1, -1) 
                                                         + np.array(posy).reshape(-1, 1) * last_estimations_v.reshape(1, -1)))
        
        U, _, Vh = np.linalg.svd(np.einsum('mnp,qp->mnq', A, Aref.conj()))
        V = Vh.conj().transpose(0, 2, 1)
        
        lowers_u = np.maximum(-np.ones((d,)), last_estimations_u-1/(2*iteration**p))
        uppers_u = np.minimum(np.ones((d,)), last_estimations_u+1/(2*iteration**p))
        lowers_v = np.maximum(-np.ones((d,)), last_estimations_v-1/(2*iteration**p))
        uppers_v = np.minimum(np.ones((d,)), last_estimations_v+1/(2*iteration**p))

        factor_u = 2 * np.pi / 3e8 * (ref_freq * posx.reshape(1, 1, -1) - freqs.reshape(-1, 1, 1) * posx.reshape(1, -1, 1))
        factor_v = 2 * np.pi / 3e8 * (ref_freq * posy.reshape(1, 1, -1) - freqs.reshape(-1, 1, 1) * posy.reshape(1, -1, 1))

        Qu = np.divide(np.exp(1j * np.expand_dims(factor_u, 0) * uppers_u.reshape(-1, 1, 1, 1)) 
                     - np.exp(1j * np.expand_dims(factor_u, 0) * lowers_u.reshape(-1, 1, 1, 1)),
                      1j * factor_u,
                      out=np.tile((uppers_u - lowers_u).reshape(-1, 1, 1, 1), ((1,)+factor_u.shape)).astype(np.complex64), 
                      where=(factor_u != 0))

        Qv = np.divide(np.exp(1j * np.expand_dims(factor_v, 0) * uppers_v.reshape(-1, 1, 1, 1)) 
                     - np.exp(1j * np.expand_dims(factor_v, 0) * lowers_v.reshape(-1, 1, 1, 1)),
                      1j * factor_v,
                      out=np.tile((uppers_v - lowers_v).reshape(-1, 1, 1, 1), ((1,)+factor_v.shape)).astype(np.complex64), 
                      where=(factor_v != 0))
        
        # Qu = (np.exp(1j * np.expand_dims(factor_u, 0) * uppers_u.reshape(-1, 1, 1, 1)) 
        #     - np.exp(1j * np.expand_dims(factor_u, 0) * lowers_u.reshape(-1, 1, 1, 1))) / (1j* np.expand_dims(factor_u, 0))
        # Qu[:, factor_u==0.] = (uppers_u - lowers_u)[:, None]

        # Qv = (np.exp(1j * np.expand_dims(factor_v, 0) * uppers_v.reshape(-1, 1, 1, 1)) 
        #     - np.exp(1j * np.expand_dims(factor_v, 0) * lowers_v.reshape(-1, 1, 1, 1))) / (1j * np.expand_dims(factor_v, 0))
        # Qv[:, factor_v==0.] = (uppers_v - lowers_v)[:, None]
        
        Q = np.sum(Qu*Qv, axis=0)
        
        F, _, Gh = np.linalg.svd(V[:, :, d:].conj().transpose(0, 2, 1) @ Q.conj().transpose(0, 2, 1) @ U[:, :, d:])

        T = V[:, :, :d] @ U[:, :, :d].conj().transpose(0, 2, 1) + V[:, :, d:] @ F @ Gh @ U[:, :, d:].conj().transpose(0, 2, 1)

        return T

    def estimate_doa_spatial_fft_robust_autofocusing(self, Rx, freqs, nbSources, ref_freq, fft_vertical_size, fft_horizontal_size, nbIterations):

        Tinit = self.get_focusing_rcsm_init(freqs, ref_freq)
        Rx_focused = np.sum(Tinit @ Rx @ Tinit.conj().transpose(0, 2, 1), axis=0)

        coarray_covariance = {}
        for pos, indices in self.pos_coarray_dict.items():
            indices = np.array(indices)
            values = np.mean(Rx_focused[indices[:, 0], indices[:, 1]])
            coarray_covariance[pos] = values

        R_padded = np.zeros((2*fft_vertical_size+1, 2*fft_horizontal_size+1), dtype=np.complex64)
        for pos in self.pos_coarray_dict.keys():
            R_padded[pos[1]+fft_vertical_size, pos[0]+fft_horizontal_size] = coarray_covariance[pos]

        R_spatial_fft = np.abs(fft(fft(R_padded, axis=0), axis=1))

        top_peaks_init = get_top_peaks(R_spatial_fft, nbSources)
        I = 3e8 / (self.min_distance * ref_freq) * ((top_peaks_init[:, 1] >= fft_horizontal_size + 1) 
                                                  - (top_peaks_init[:, 1] / (2 * fft_horizontal_size + 1)))
        J = 3e8 / (self.min_distance * ref_freq) * ((top_peaks_init[:, 0] >= fft_horizontal_size + 1) 
                                                  - (top_peaks_init[:, 0] / (2 * fft_horizontal_size + 1)))
        sol = np.linalg.solve(self.shear.T, np.stack((I, J), axis=0))
        last_estimations_u = sol[0]
        last_estimations_v = sol[1]

        for iter in range(1, nbIterations):

            Titer = self.get_focusing_rcsm_iter(freqs, ref_freq, last_estimations_u, last_estimations_v, iter, 1.5)
            Rx_focused = np.mean(Titer @ Rx @ Titer.conj().transpose(0, 2, 1), axis=0)

            coarray_covariance = {}
            for pos, indices in self.pos_coarray_dict.items():
                indices = np.array(indices)
                values = np.mean(Rx_focused[indices[:, 0], indices[:, 1]])
                coarray_covariance[pos] = values

            R_padded = np.zeros((2*fft_vertical_size+1, 2*fft_horizontal_size+1), dtype=np.complex64)
            for pos in self.pos_coarray_dict.keys():
                R_padded[pos[1]+fft_vertical_size, pos[0]+fft_horizontal_size] = coarray_covariance[pos]

            R_spatial_fft = np.abs(fft(fft(R_padded, axis=0), axis=1))

            top_peaks_init = get_top_peaks(R_spatial_fft, nbSources)
            I = 3e8 / (self.min_distance * ref_freq) * ((top_peaks_init[:, 1] >= fft_horizontal_size + 1) 
                                                      - (top_peaks_init[:, 1] / (2 * fft_horizontal_size + 1)))
            J = 3e8 / (self.min_distance * ref_freq) * ((top_peaks_init[:, 0] >= fft_horizontal_size + 1) 
                                                      - (top_peaks_init[:, 0] / (2 * fft_horizontal_size + 1)))
            sol = np.linalg.solve(self.shear.T, np.stack((I, J), axis=0))
            last_estimations_u = sol[0]
            last_estimations_v = sol[1]

        estimated_theta = np.arcsin(np.clip(np.sqrt(last_estimations_u ** 2 + last_estimations_v ** 2), -1, 1))
        estimated_phi = np.arctan2(last_estimations_v, last_estimations_u)
        estimated_phi = np.where(estimated_phi < 0, estimated_phi + np.pi * 2, estimated_phi)
        estimated_phi = np.where(estimated_phi > np.pi, 2 * np.pi - estimated_phi, estimated_phi)

        return estimated_phi, estimated_theta
    
    def estimate_doa_music_robust_autofocusing(self, Rx, freqs, nbSources, ref_freq, nbIterations):

        Tinit = self.get_focusing_rcsm_init(freqs, ref_freq)
        Rx_focused = np.sum(Tinit @ Rx @ Tinit.conj().transpose(0, 2, 1), axis=0)

        _, vecs = np.linalg.eigh(Rx_focused)
        En = vecs[:, :-nbSources]
        array_manifold = self.array_manifold(ref_freq)
        spectrum = 1 / np.linalg.norm(np.einsum('mi,mjk->ijk', np.conj(En), array_manifold), axis=0) ** 2

        top_peaks_init = get_top_peaks(spectrum, nbSources)
        estimated_phi, estimated_theta = self.phi_space[top_peaks_init[:, 0]], self.theta_space[top_peaks_init[:, 1]]

        for iter in range(1, nbIterations):

            last_estimations_u = np.cos(estimated_phi) * np.sin(estimated_theta)
            last_estimations_v = np.sin(estimated_phi) * np.sin(estimated_theta)
            Titer = self.get_focusing_rcsm_iter(freqs, ref_freq, last_estimations_u, last_estimations_v, iter, 1.5)
            Rx_focused = np.mean(Titer @ Rx @ Titer.conj().transpose(0, 2, 1), axis=0)

            _, vecs = np.linalg.eigh(Rx_focused)
            En = vecs[:, :-nbSources]
            spectrum = 1 / np.linalg.norm(np.einsum('mi,mjk->ijk', np.conj(En), array_manifold), axis=0) ** 2

            top_peaks_init = get_top_peaks(spectrum, nbSources)
            estimated_phi, estimated_theta = self.phi_space[top_peaks_init[:, 0]], self.theta_space[top_peaks_init[:, 1]]

        return estimated_phi, estimated_theta


class UniformRectangularArray(ArrayModelAbstractClass):

    def __init__(self, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):

        self.min_distance = min_distance
        self.size_horizontal = size_horizontal
        self.size_vertical = size_vertical
        self.nbSensors = size_horizontal * size_vertical
        self.shear = shear

        self.x = np.tile(np.arange(0, size_horizontal, 1), size_vertical).tolist()
        self.y = np.repeat(np.arange(0, size_vertical, 1), size_horizontal).tolist()


class NestedArray2D(ArrayModelAbstractClass):

    def __init__(self, 
                 min_distance: float, 
                 levels_horizontal: list[int], 
                 levels_vertical: list[int], 
                 shear = np.array([[1, 0], [0, 1]])):
        
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

    def __init__(self, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):

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

    def __init__(self, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):

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

    def __init__(self, 
                 min_distance: float, 
                 size_horizontal: int, 
                 size_vertical: int, 
                 shear = np.array([[1, 0], [0, 1]])):

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


class HexagonalArray(ArrayModelAbstractClass):

    def __init__(self, 
                 min_distance: float, 
                 shear = np.array([[1, 1/2], [0, sqrt(3)/2]])):

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


def generate_ofdm_sources(nbSources, nbSnapshots, fSampling, fmin, fmax, nbSubcarrier=256):
    
    t = np.arange(nbSnapshots) / fSampling
    f = np.linspace(0, fmax - fmin, nbSubcarrier, endpoint=False)
    fc = np.random.rand(nbSources) * (fmax - fmin) / nbSubcarrier + fmin
    symbols = (np.random.randn(nbSources, nbSubcarrier) + 1j * np.random.randn(nbSources, nbSubcarrier)) / sqrt(2)
    sources = symbols @ np.exp(1j * 2 * np.pi * f.reshape(-1, 1) * t.reshape(1, -1)) / nbSubcarrier
    sources = sources * np.exp(1j * 2 * np.pi * fc.reshape(-1, 1) * t.reshape(1, -1))
    
    return sources


def get_top_peaks(mat, nbPeaks):

    neighborhood = maximum_filter(mat, size=3, mode='constant', cval=-np.inf)
    peaks_mask = (mat == neighborhood)
    for shift in [(-1,0), (1,0), (0,-1), (0,1)]:
        shifted = np.roll(mat, shift, axis=(0,1))
        mask = np.ones_like(mat, dtype=bool)
        if shift[0] == -1:
            mask[0,:] = False
        elif shift[0] == 1:
            mask[-1,:] = False
        if shift[1] == -1:
            mask[:,0] = False
        elif shift[1] == 1:
            mask[:,-1] = False
        peaks_mask &= (mat > shifted) | ~mask  

    peak_indices = np.argwhere(peaks_mask)
    peak_values = mat[peaks_mask]
    sorted_indices = np.argsort(-peak_values)
    top_peaks = [(peak_indices[i]) for i in sorted_indices[:nbPeaks]]
    top_peaks = np.array(top_peaks, dtype=int)
    return top_peaks 


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