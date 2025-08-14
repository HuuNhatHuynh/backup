#!/bin/bash

MC=200
nbSources=5
sizeSpatialFFT=1024
nbIterations=5
snr_values=(0 5 10 15 20 25)

mkdir results_robust_autofocusing

for snr in "${snr_values[@]}"; do
    python3 monte_carlo_robust_autofocusing.py --snr "$snr" --mc "$MC" --nbIterations "$nbIterations" --nbSources "$nbSources" --sizeSpatialFFT "$sizeSpatialFFT" >> results_robust_autofocusing/timing_results.txt &
done

wait