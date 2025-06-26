import argparse
import csv
import os
import random
import numpy as np

from qiskit.quantum_info import Statevector, state_fidelity, random_statevector
from evolution_engine import Population
from known_states import (
    get_ghz_statevector,
    get_bell_statevector,
    get_w_statevector,
    get_w_gate_sequence,
    get_ghz_gate_sequence,
    get_bell_gate_sequence
)
from parameter_optimizer import optimize_parameters
from prune import prune_circuit
from circuit_representation import QuantumCircuitCandidate
from fitness_evaluation import compute_entropy
from plotting import plot_fidelity_vs_depth

def get_target_and_seed(target_name, num_qubits):
    if target_name == "ghz":
        return get_ghz_statevector(num_qubits), get_ghz_gate_sequence(num_qubits)
    elif target_name == "bell":
        return get_bell_statevector(), get_bell_gate_sequence()
    elif target_name == "w":
        return get_w_statevector(num_qubits), get_w_gate_sequence()
    elif target_name == "haar":
        return random_statevector(2**num_qubits), None
    else:
        raise ValueError(f"Unknown target: {target_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="ghz", help="Target state: bell, ghz, w, haar")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.0, help="Entropy reward factor")
    parser.add_argument("--output", type=str, default="logs.csv")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    NUM_QUBITS = 3
    target, seed = get_target_and_seed(args.target, NUM_QUBITS)

    pop = Population(
        size=args.pop_size,
        num_qubits=NUM_QUBITS,
        max_depth=args.max_depth
    )
    pop.initialize(seed_gate_sequence=seed)

    os.makedirs("output", exist_ok=True)
    log_path = os.path.join("output", args.output)

    with open(log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "fidelity", "depth", "gate_count", "entropy"])

        for gen in range(args.generations):
            best_fitness = pop.evolve(target, crossover_rate=0.5, gamma=args.gamma)
            best_ind = pop.evaluate(target, gamma=args.gamma)[0][1]

            qc = best_ind.to_qiskit_circuit()
            sv = Statevector.from_instruction(qc)
            fidelity = state_fidelity(sv, target)
            depth = qc.depth()
            gate_count = best_ind.gate_count()
            entropy = compute_entropy(sv, NUM_QUBITS)

            writer.writerow([gen + 1, fidelity, depth, gate_count, entropy])
            print(f"🧬 Generation {gen+1:02d} | 🏆 Fidelity: {fidelity:.5f} | Depth: {depth} | Entropy: {entropy:.3f}")

    # Final circuit optimization + pruning
    best_individual = pop.evaluate(target, gamma=args.gamma)[0][1]
    optimized = optimize_parameters(best_individual, target, steps=500, lr=0.02)
    pruned = prune_circuit(optimized, target)

    qc = pruned.to_qiskit_circuit()
    print(qc.draw('text'))
    print("🎯 Final Fidelity:", state_fidelity(Statevector.from_instruction(qc), target))
    print("📏 Final Gate Count:", len(pruned.gate_sequence))

    plot_fidelity_vs_depth(log_path, title=f"Evolution for {args.target.upper()}")

if __name__ == "__main__":
    main()
