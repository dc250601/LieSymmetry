#!/bin/bash

module load tensorflow/2.12.0

# # Launch 16 tasks in parallel
# for i in $(seq 0 15); do
#     echo "Launching task index $i"
#     srun --exclusive -n1 -N1 python3 so_script.py --order 3 --store_idx "$i" &
# done

# wait

# for i in $(seq 0 15); do
#     echo "Launching task index $i"
#     srun --exclusive -n1 -N1 python3 so_script.py --order 4 --store_idx "$i" &
# done

# wait

for i in $(seq 0 15); do
    echo "Launching task index $i"
    srun --exclusive -n1 -N1 python3 so_script.py --order 5 --store_idx "$i" &
done

wait
echo "All GPU tasks complete!"
