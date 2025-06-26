import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity
from fitness_evaluation import compute_penalty

def extract_angles(gate_sequence):
    """Flatten list of gate angles"""
    angles = []
    for gate in gate_sequence:
        if 'angle' in gate:
            angles.append(gate['angle'])
    return np.array(angles)

def inject_angles(gate_sequence, new_angles):
    """Replace angle values back into gate sequence"""
    i = 0
    for gate in gate_sequence:
        if 'angle' in gate:
            gate['angle'] = new_angles[i]
            i += 1
    return gate_sequence

def optimize_parameters(candidate, target_statevector, alpha=0.05, beta=0.01, steps=100, lr=0.1):
    """Gradient-free parameter optimizer (simple hill climbing)"""
    angles = extract_angles(candidate.gate_sequence)
    if angles.size == 0:
        return candidate  # Nothing to optimize

    best_angles = angles.copy()
    best_fitness = state_fidelity(Statevector.from_instruction(candidate.to_qiskit_circuit()), target_statevector) - compute_penalty(candidate, alpha, beta)

    for _ in range(steps):
        perturbation = np.random.normal(0, lr, size=angles.shape)
        trial_angles = best_angles + perturbation
        inject_angles(candidate.gate_sequence, trial_angles)

        try:
            sv = Statevector.from_instruction(candidate.to_qiskit_circuit())
            fidelity = state_fidelity(sv, target_statevector)
            penalty = compute_penalty(candidate, alpha, beta)
            fitness = fidelity - penalty

            if fitness > best_fitness:
                best_angles = trial_angles
                best_fitness = fitness
        except:
            continue

    inject_angles(candidate.gate_sequence, best_angles)
    return candidate
