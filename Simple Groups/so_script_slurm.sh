#!/bin/bash

module load tensorflow/2.12.0

python3 so_script.py 3 0&
python3 so_script.py 4 0&
python3 so_script.py 5 0&

python3 so_script.py 6 1&
python3 so_script.py 7 1&

python3 so_script.py 8 2&
python3 so_script.py 9 2&

python3 so_script.py 10 3&

wait