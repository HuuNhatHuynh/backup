import json, argparse, time
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser(description='Monte Carlo simulation for antenna arrays')
parser.add_argument('--mc', type=int, default=100, help='Number of Monte Carlo trials')
parser.add_argument('--snr', type=float, default=20, help='SNR of signals')
parser.add_argument('--nbSources', type=int, default=5, help='Number of sources')
parser.add_argument('--sizeSpatialFFT', type=int, default=512, help='Size of spatial FFT')
parser.add_argument('--nbIterations', type=int, default=10, help='Number of iterations of wideband DoA estimation algorithms')
args = parser.parse_args()

snr = args.snr
mc = args.mc
nbSources = args.nbSources
nbChunks = 50
chunkSize = 100
fCarrier = 5000.0
bandwidth = 2000.0
nbSnapshots = nbChunks*chunkSize
fmin = fCarrier-bandwidth/2
fmax = fCarrier+bandwidth/2
fSampling = int(3*fmax)
min_distance = (3e8/fmax+1)/2
ref_freq = 5000.0 
nbIterations = args.nbIterations
sizeTemporalFFT = 256
sizeHorizontalFFT = args.sizeSpatialFFT
sizeVerticalFFT = args.sizeSpatialFFT

rmspe_func = RMSPE(nbSources)

array_dict = {}

# array_dict["Hexagonal Array"] = HexagonalArray(min_distance=min_distance)
array_dict["Nested Array"] = NestedArray2D(min_distance=min_distance, levels_horizontal=[5, 5], levels_vertical=[5, 5])
array_dict["Open Box Array"] = OpenBoxArray(min_distance=min_distance, size_horizontal=35, size_vertical=34)
# array_dict["Half Open Box Array"] = HalfOpenBoxArray(min_distance=min_distance, size_horizontal=35, size_vertical=34)
# array_dict["Half Open Box Array 2"] = HalfOpenBoxArray2(min_distance=min_distance, size_horizontal=35, size_vertical=34)

for array in array_dict.values():
    array.build_coarray()
    array.build_array_manifold(ref_freq)

rmspe = {}
time_ratio = {}

for name, array in array_dict.items(): 
    rmspe[name+" - FFT"] = []
    rmspe[name+" - MUSIC"] = []
    time_ratio[name] = []

for _ in tqdm(range(mc)):
    
    phi_true = generate_angles(nbSources, 0, np.pi)
    theta_true = generate_angles(nbSources, 0, np.pi/2)
    sources = generate_ofdm_sources(nbSources, 16384, fSampling, fmin, fmax)
    
    for name, array in array_dict.items():
        
        x = array.generate_broadband_data(sources, phi_true, theta_true, snr, nbSnapshots, fSampling)
        Rx, freqs = array.get_covariances(x, nbChunks, fmin, fmax, fSampling, sizeTemporalFFT)

        start_fft = time.perf_counter()
        estimated_phi_fft, estimated_theta_fft = array.estimate_doa_spatial_fft_robust_autofocusing(Rx, freqs, nbSources, ref_freq, sizeVerticalFFT, sizeHorizontalFFT, nbIterations)
        end_fft = time.perf_counter()
        rmspe[name+" - FFT"].append(rmspe_func.calculate(estimated_phi_fft, estimated_theta_fft, phi_true, theta_true))

        start_music = time.perf_counter()
        estimated_phi_music, estimated_theta_music = array.estimate_doa_music_robust_autofocusing(Rx, freqs, nbSources, ref_freq, nbIterations)
        end_music = time.perf_counter()
        rmspe[name+" - MUSIC"].append(rmspe_func.calculate(estimated_phi_music, estimated_theta_music, phi_true, theta_true))

        time_ratio[name].append((end_music - start_music) / (end_fft - start_fft))

result = {}
for key, val in rmspe.items(): 
    result[key] = sum(val) / mc

with open("results_robust_autofocusing/results_{}_db.json".format(snr), "w") as f:
    json.dump(result, f, indent=4)

for name in time_ratio.keys():
    print("Ratio of execution time between FFT and MUSIC of "+name+" at {} dB: {}".format(snr, sum(time_ratio[name]) / mc))
print("\n")