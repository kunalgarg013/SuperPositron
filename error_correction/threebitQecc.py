from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt  

# Define a 3-qubit Quantum Register and 1-bit Classical Register
qreg = QuantumRegister(3, 'q')
creg = ClassicalRegister(1, 'c')
qc = QuantumCircuit(qreg, creg)

# Encode a bit-flip error correction code (assume starting in |0⟩ state)
qc.h(0)        # Hadamard for superposition
qc.cx(0, 1)    # Copy state to qubit 1
qc.cx(0, 2)    # Copy state to qubit 2

# Introduce an error (simulate a bit-flip on qubit 1)
qc.x(1)  # Flip qubit 1

# Error Correction: Detect & Correct
qc.cx(1, 0)
qc.cx(2, 0)

# Measurement
qc.measure(0, 0)

# Run the circuit on AerSimulator
simulator = AerSimulator()
result = simulator.run(qc).result()
counts = result.get_counts()

# Visualize results
print(counts)
plot_histogram(counts)
plt.show()  # Ensure the plot is displayed
