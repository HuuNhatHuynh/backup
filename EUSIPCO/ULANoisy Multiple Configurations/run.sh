#!/bin/bash

python3 train.py --snr -10 &
python3 train.py --snr -5 &
python3 train.py --snr 0 &
python3 train.py --snr 5 &
python3 train.py --snr 10 &
python3 train.py --snr 15 &
python3 train.py --snr 20 &

wait

python3 test.py

echo "All training processes have completed"