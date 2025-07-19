#!/bin/bash

module load tensorflow/2.12.0

python3 so_convergence_test.py 5 0 0&
python3 so_convergence_test.py 5 0 1&
python3 so_convergence_test.py 5 0 2&
python3 so_convergence_test.py 5 0 3&
python3 so_convergence_test.py 5 0 4&

python3 so_convergence_test.py 5 1 5&
python3 so_convergence_test.py 5 1 6&
python3 so_convergence_test.py 5 1 7&
python3 so_convergence_test.py 5 1 8&
python3 so_convergence_test.py 5 1 9&

python3 so_convergence_test.py 5 2 10&
python3 so_convergence_test.py 5 2 11&
python3 so_convergence_test.py 5 2 12&
python3 so_convergence_test.py 5 2 13&
python3 so_convergence_test.py 5 2 14&

python3 so_convergence_test.py 5 3 15&
python3 so_convergence_test.py 5 3 16&
python3 so_convergence_test.py 5 3 17&
python3 so_convergence_test.py 5 3 18&
python3 so_convergence_test.py 5 3 19&
wait