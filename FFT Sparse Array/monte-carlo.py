import json, argparse
import numpy as np 
import concurrent.futures
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser(description='Monte Carlo simulation for antenna arrays')
parser.add_argument('--t', type=int, default=200, help='Number of snapshots (default: 200)')
parser.add_argument('--mc', type=int, default=1000, help='Number of Monte Carlo trials (default: 1000)')
args = parser.parse_args()

lamda = 0.2
min_distance = 0.1 
t = args.t
nbSources = 5
SNRs = [-10, -5, 0, 5, 10, 15, 20]
mc = args.mc

rmspe_func = RMSPE(nbSources)

list_arrays = []

HA = HexagonalArray(lamda=lamda, min_distance=min_distance)
list_arrays.append(("Hexagonal Array", HA, True, True))

NA = NestedArray2D(lamda=lamda, min_distance=min_distance, levels_horizontal=[5, 5], levels_vertical=[5, 5])
list_arrays.append(("Nested Array", NA, True, True))

OB = OpenBoxArray(lamda=lamda, min_distance=min_distance, size_horizontal=35, size_vertical=34)
list_arrays.append(("Open Box Array", OB, True, True))

HOB = HalfOpenBoxArray(lamda=lamda, min_distance=min_distance, size_horizontal=35, size_vertical=34)
list_arrays.append(("Half Open Box Array", HOB, True, True))

HOB2 = HalfOpenBoxArray2(lamda=lamda, min_distance=min_distance, size_horizontal=35, size_vertical=34)
list_arrays.append(("Half Open Box Array 2", HOB2, True, True))

URA = UniformRectangularArray(lamda=lamda, min_distance=min_distance, size_horizontal=35, size_vertical=34)
list_arrays.append(("Uniform Rectangular Array", URA, True, False))

for _, array, _, _ in list_arrays:
    array.build_coarray()
    array.build_array_manifold()


def run_trial(snr):

    rmspe = {}
    for name, _, fft, music in list_arrays: 
        if fft: rmspe[name+" - FFT"] = []
        if music: rmspe[name+" - MUSIC"] = []

    for _ in tqdm(range(mc)):
        phi_true = generate_angles(nbSources, 0, np.pi)
        theta_true = generate_angles(nbSources, 0, np.pi/2)
        S = (np.random.randn(nbSources, t) + 1j * np.random.randn(nbSources, t)) / sqrt(2)
        for name, array, fft, music in list_arrays:
            noise = (np.random.randn(array.nbSensors, t) + 1j * np.random.randn(array.nbSensors, t)) / sqrt(2) * sqrt(nbSources) * 10 ** (-snr/20)
            X = array.get_steering_vector(phi_true, theta_true) @ S + noise
            if fft:
                estimated_phi_fft, estimated_theta_fft = array.estimate_doa_fft(X, nbSources)
                rmspe[name+" - FFT"].append(rmspe_func.calculate(estimated_phi_fft, estimated_theta_fft, phi_true, theta_true))
            if music:
                estimated_phi_music, estimated_theta_music = array.estimate_doa_music(X, nbSources)
                rmspe[name+" - MUSIC"].append(rmspe_func.calculate(estimated_phi_music, estimated_theta_music, phi_true, theta_true))

    result = {}
    for key, val in rmspe.items(): 
        result[key] = sum(val) / mc

    return result

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(run_trial, SNRs))

# plot_data = {}
# for key in results[0].keys(): plot_data[key] = []

# for idx in range(len(SNRs)):
#     for key in plot_data.keys():
#         plot_data[key].append(results[idx][key])

# with open("results_t{}.json".format(t, mc), "w") as f:
#     json.dump(plot_data, f, indent=4)