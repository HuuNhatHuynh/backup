#!/bin/bash

MC=10
nbSources=5
sizeSpatialFFT=1024
snr_values=(0 5 10 15 20 25)

mkdir results_autofocusing

for snr in "${snr_values[@]}"; do
    python3 monte_carlo_autofocusing.py --snr "$snr" --mc "$MC" --nbSources "$nbSources" --sizeSpatialFFT "$sizeSpatialFFT" >> results_autofocusing/timing_results.txt &
done

wait