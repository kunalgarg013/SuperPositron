from circuit_representation import QuantumCircuitCandidate
from known_states import get_bell_statevector
from evolution_engine import ghz_circuit_gate_sequence  # reuse structure
from parameter_optimizer import optimize_parameters
from prune import prune_circuit
from qiskit.quantum_info import Statevector, state_fidelity

from known_states import get_w_statevector

# Prepare W target state
target = get_w_statevector(3)

# Use GHZ structure as a warm start
w = QuantumCircuitCandidate(3, 10)
w.gate_sequence = ghz_circuit_gate_sequence(3)

# Optimize
w_optimized = optimize_parameters(w, target, steps=300, lr=0.05)
w_pruned = prune_circuit(w_optimized, target)

# Report
qc = w_pruned.to_qiskit_circuit()
print(qc.draw('text'))
print("🎯 Fidelity (W):", state_fidelity(Statevector.from_instruction(qc), target))
print("📏 Gate Count (W):", len(w_pruned.gate_sequence))