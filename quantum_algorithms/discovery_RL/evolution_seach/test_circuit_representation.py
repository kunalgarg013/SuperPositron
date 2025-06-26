from circuit_representation import QuantumCircuitCandidate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import random_statevector
from fitness_evaluation import compute_fitness

def test_basic_init_and_mutation():
    print("🔧 Creating random candidate...")
    candidate = QuantumCircuitCandidate(num_qubits=3, max_depth=10)
    candidate.random_initialize()
    candidate.describe()

    print("\n🧬 Applying mutation...")
    candidate.mutate()
    candidate.describe()

    print("\n🎯 Converting to Qiskit circuit...")
    qc = candidate.to_qiskit_circuit()
    print(qc.draw('text'))

    print(f"📏 Estimated gate count: {candidate.gate_count()}")
    print(f"📏 Estimated depth (naive): {candidate.depth_estimate()}")

    print("\n🧠 Simulating final state using Statevector...")
    try:
        sv = Statevector.from_instruction(qc)
        print("✅ Final statevector computed.")
        print(sv)
    except Exception as e:
        print("❌ Failed to simulate circuit:", e)

    # Create a random 3-qubit target state
    target = random_statevector(2 ** 3)

    fitness = compute_fitness(candidate, target)
    print(f"\n🎯 Fitness vs random target state: {fitness:.5f}")

if __name__ == "__main__":
    test_basic_init_and_mutation()
