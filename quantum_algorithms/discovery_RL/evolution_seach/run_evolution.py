from qiskit.quantum_info import Statevector, state_fidelity
from evolution_engine import Population
from known_states import get_ghz_statevector, get_bell_statevector, get_w_statevector, get_w_gate_sequence
from parameter_optimizer import optimize_parameters
from prune import prune_circuit
from circuit_representation import QuantumCircuitCandidate

NUM_QUBITS = 3
MAX_DEPTH = 30
POP_SIZE = 60
GENERATIONS = 10

# Choose your target state
target = get_w_statevector(NUM_QUBITS)

# Seed W circuit
seed = QuantumCircuitCandidate(3, 10)
seed.gate_sequence = get_w_gate_sequence()

# Initialize population
pop = Population(size=POP_SIZE, num_qubits=NUM_QUBITS, max_depth=MAX_DEPTH)
pop.initialize()

# Evolution loop
for gen in range(GENERATIONS):
    best_fitness = pop.evolve(target, crossover_rate=0.5)
    print(f"🧬 Generation {gen+1:02d} | 🏆 Best Fitness: {best_fitness:.5f}")

# After evolution: take best candidate, optimize and prune
best_individual = pop.evaluate(target)[0][1]
optimized = optimize_parameters(best_individual, target, steps=500, lr=0.02)
pruned = prune_circuit(optimized, target)

# Final results
qc = pruned.to_qiskit_circuit()
print(qc.draw('text'))
print("🎯 Final Fidelity:", state_fidelity(Statevector.from_instruction(qc), target))
print("📏 Final Gate Count:", len(pruned.gate_sequence))
