import os
import csv
import random
import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity, random_statevector

from evolution_engine import Population
from known_states import (
    get_ghz_statevector,
    get_bell_statevector,
    get_w_statevector,
    get_ghz_gate_sequence,
    get_bell_gate_sequence,
    get_w_gate_sequence
)
from parameter_optimizer import optimize_parameters
from prune import prune_circuit
from fitness_evaluation import compute_entropy


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

def run_experiment(seed, target_name, num_qubits, generations, pop_size, max_depth, gamma):
    random.seed(seed)
    np.random.seed(seed)

    target, seed_sequence = get_target_and_seed(target_name, num_qubits)
    pop = Population(size=pop_size, num_qubits=num_qubits, max_depth=max_depth)
    pop.initialize(seed_gate_sequence=seed_sequence)

    best_fid = -1
    best_ind = None

    for gen in range(generations):
        fid = pop.evolve(target, crossover_rate=0.5, gamma=gamma)
        top_ind = pop.evaluate(target, gamma=gamma)[0][1]
        sv = Statevector.from_instruction(top_ind.to_qiskit_circuit())
        fidelity = state_fidelity(sv, target)
        if fidelity > best_fid:
            best_fid = fidelity
            best_ind = top_ind

    optimized = optimize_parameters(best_ind, target, steps=300, lr=0.01)
    pruned = prune_circuit(optimized, target)
    qc = pruned.to_qiskit_circuit()
    final_sv = Statevector.from_instruction(qc)
    fidelity = state_fidelity(final_sv, target)
    entropy = compute_entropy(final_sv, num_qubits)
    depth = qc.depth()
    gate_count = pruned.gate_count()

    return {
        "seed": seed,
        "fidelity": fidelity,
        "entropy": entropy,
        "depth": depth,
        "gate_count": gate_count
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="haar")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--pop-size", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="output/seed_sweep_log.csv")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    with open(args.output, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "fidelity", "entropy", "depth", "gate_count"])

        for s in range(args.num_seeds):
            result = run_experiment(
                seed=s,
                target_name=args.target,
                num_qubits=3,
                generations=args.generations,
                pop_size=args.pop_size,
                max_depth=args.max_depth,
                gamma=args.gamma
            )
            print(f"🌱 Seed {s:02d} | Fidelity: {result['fidelity']:.5f} | Depth: {result['depth']} | Entropy: {result['entropy']:.3f}")
            writer.writerow([
                result["seed"],
                result["fidelity"],
                result["entropy"],
                result["depth"],
                result["gate_count"]
            ])


if __name__ == "__main__":
    main()
