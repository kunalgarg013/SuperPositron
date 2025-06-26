from qiskit.quantum_info import Statevector, state_fidelity

def compute_statevector(candidate):
    """Simulate the circuit and return its statevector."""
    try:
        qc = candidate.to_qiskit_circuit()
        sv = Statevector.from_instruction(qc)
        return sv
    except Exception as e:
        print(f"❌ Error simulating statevector: {e}")
        return None

def compute_fidelity(candidate, target_statevector):
    """Compute fidelity between candidate circuit and target state."""
    sv = compute_statevector(candidate)
    if sv is None:
        return 0.0
    return state_fidelity(sv, target_statevector)

def compute_penalty(candidate, alpha=0.05, beta=0.01):
    """Compute penalty based on gate depth and total gate count."""
    return alpha * candidate.depth_estimate() + beta * candidate.gate_count()

def compute_fitness(candidate, target_statevector, alpha=0.05, beta=0.01):
    """Compute total fitness: Fidelity - Penalty"""
    fidelity = compute_fidelity(candidate, target_statevector)
    penalty = compute_penalty(candidate, alpha, beta)
    return fidelity - penalty
