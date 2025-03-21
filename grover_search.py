from qiskit import QuantumCircuit, transpile  
from qiskit_aer import Aer  
from qiskit_aer.backends import AerSimulator  

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def grover_oracle(qc, target_state):
    """Applies the oracle for marking |101⟩."""
    qc.x([0, 2])  # Flip qubits to match |101⟩
    qc.h(2)  # Control on qubit 2
    qc.ccx(0, 1, 2)  # Multi-controlled NOT (Toffoli gate)
    qc.h(2)
    qc.x([0, 2])  # Unflip back

def diffusion_operator(qc):
    """Applies the diffusion operator."""
    qc.h([0, 1, 2])  # Apply Hadamard to all
    qc.x([0, 1, 2])  # Flip qubits
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])  # Unflip
    qc.h([0, 1, 2])  # Final Hadamard

# Step 1: Initialize circuit with 3 qubits and 3 classical bits
qc = QuantumCircuit(3, 3)

# Step 2: Create uniform superposition
qc.h([0, 1, 2])

# Step 3: Apply Grover's Oracle
grover_oracle(qc, "101")

# Step 4: Apply Diffusion Operator
diffusion_operator(qc)

# Step 5: Measure
qc.measure([0, 1, 2], [0, 1, 2])

# Simulate the circuit
simulator = AerSimulator()
tqc = transpile(qc, simulator)
# qobj = assemble(tqc)
result = simulator.run(tqc, shots=10000000).result()
counts = result.get_counts()

# Plot the results
qc.draw('mpl')
plot_histogram(counts)
plt.show()
