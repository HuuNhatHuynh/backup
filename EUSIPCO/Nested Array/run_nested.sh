#!/bin/bash

python3 train_nested.py --snr -10 &
python3 train_nested.py --snr -5 &
python3 train_nested.py --snr 0 &
python3 train_nested.py --snr 5 &
python3 train_nested.py --snr 10 &
python3 train_nested.py --snr 15 &
python3 train_nested.py --snr 20 &

wait

echo "All training processes have completed"

python3 select_nested.py --snr -10 &
python3 select_nested.py --snr -5 &
python3 select_nested.py --snr 0 &
python3 select_nested.py --snr 5 &
python3 select_nested.py --snr 10 &
python3 select_nested.py --snr 15 &
python3 select_nested.py --snr 20 &

wait

echo "All selection processes have completed"