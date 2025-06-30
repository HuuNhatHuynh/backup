#!/bin/bash

python3 train_uca.py --snr -10 &
python3 train_uca.py --snr -5 &
python3 train_uca.py --snr 0 &
python3 train_uca.py --snr 5 &
python3 train_uca.py --snr 10 &
python3 train_uca.py --snr 15 &
python3 train_uca.py --snr 20 &

wait

echo "All training processes have completed"

python3 select_uca.py --snr -10 &
python3 select_uca.py --snr -5 &
python3 select_uca.py --snr 0 &
python3 select_uca.py --snr 5 &
python3 select_uca.py --snr 10 &
python3 select_uca.py --snr 15 &
python3 select_uca.py --snr 20 &

wait

echo "All selection processes have completed"