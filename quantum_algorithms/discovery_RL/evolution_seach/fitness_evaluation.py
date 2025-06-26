from qiskit.quantum_info import Statevector, state_fidelity, partial_trace, entropy

def compute_statevector(candidate):
    try:
        qc = candidate.to_qiskit_circuit()
        sv = Statevector.from_instruction(qc)
        return sv
    except Exception as e:
        print(f"❌ Error simulating statevector: {e}")
        return None

def compute_fidelity(candidate, target_statevector):
    sv = compute_statevector(candidate)
    if sv is None:
        return 0.0
    return state_fidelity(sv, target_statevector)

def compute_penalty(candidate, alpha=0.05, beta=0.01):
    return alpha * candidate.depth_estimate() + beta * candidate.gate_count()

def compute_entropy(statevector, num_qubits):
    entropies = []
    for i in range(num_qubits):
        reduced = partial_trace(statevector, [j for j in range(num_qubits) if j != i])
        entropies.append(entropy(reduced))
    return sum(entropies) / num_qubits

def compute_fitness(candidate, target_statevector, alpha=0.05, beta=0.01, gamma=0.0):
    sv = compute_statevector(candidate)
    if sv is None:
        return -999.0
    fidelity = state_fidelity(sv, target_statevector)
    penalty = compute_penalty(candidate, alpha, beta)
    ent = compute_entropy(sv, candidate.num_qubits)
    return fidelity - penalty + gamma * ent
