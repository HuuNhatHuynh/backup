#!/bin/bash

python3 select_0dB.py &
python3 select_5dB.py &
python3 select_10dB.py &
python3 select_15dB.py &
python3 select_20dB.py &

wait

echo "All training processes have completed."