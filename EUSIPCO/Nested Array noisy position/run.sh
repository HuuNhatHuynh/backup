#!/bin/bash

python3 train_nested_0dB.py &
python3 train_nested_5dB.py &
python3 train_nested_10dB.py &
python3 train_nested_15dB.py &
python3 train_nested_20dB.py &

wait

echo "All training processes have completed."