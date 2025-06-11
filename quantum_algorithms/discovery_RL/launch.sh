#!/bin/bash

echo "Choose training mode:"
echo "1. Curriculum (bell → ghz → w)"
echo "2. Custom (requires config JSON)"
read -p "Enter choice [1 or 2]: " choice

if [ "$choice" == "1" ]; then
    python qiskit_rl_cuda_final.py --mode curriculum
elif [ "$choice" == "2" ]; then
    read -p "Enter path to config JSON: " config
    python qiskit_rl_cuda_final.py --mode custom --config "$config"
else
    echo "Invalid choice."
fi
