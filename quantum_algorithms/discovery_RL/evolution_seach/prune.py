from qiskit.quantum_info import Statevector, state_fidelity
from copy import deepcopy
from fitness_evaluation import compute_penalty

def prune_circuit(candidate, target_statevector, alpha=0.05, beta=0.01):
    """
    Iteratively remove gates that do not reduce fidelity.
    Stops when no further pruning is possible.
    """
    best_candidate = candidate.copy()
    best_circuit = best_candidate.to_qiskit_circuit()
    best_state = Statevector.from_instruction(best_circuit)
    best_fid = state_fidelity(best_state, target_statevector)
    best_penalty = compute_penalty(best_candidate, alpha, beta)
    best_score = best_fid - best_penalty

    changed = True
    while changed:
        changed = False
        for i in range(len(best_candidate.gate_sequence)):
            pruned = best_candidate.copy()
            del pruned.gate_sequence[i]

            try:
                sv = Statevector.from_instruction(pruned.to_qiskit_circuit())
                fid = state_fidelity(sv, target_statevector)
                penalty = compute_penalty(pruned, alpha, beta)
                score = fid - penalty

                if score >= best_score:
                    best_candidate = pruned
                    best_score = score
                    changed = True
                    break  # Restart loop after pruning
            except:
                continue  # Skip invalid gates

    return best_candidate
