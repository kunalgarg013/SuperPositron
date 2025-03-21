import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Function to encode logical qubit using 3-qubit repetition code
def encode(qc):
    qc.cx(0, 1)
    qc.cx(0, 2)

# Function to introduce an X error on a random qubit
def introduce_error(qc):
    error_qubit = np.random.choice([0, 1, 2])
    qc.x(error_qubit)
    print(f"Introduced X error on qubit {error_qubit}")

# Function to detect error using syndrome measurement
def measure_syndrome(qc):
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([1, 2], [0, 1])  # Syndrome bits stored in classical bits 0 and 1

# Function to correct error based on syndrome measurement
def correct_error(qc):
    qc.x(0).c_if(qc.cregs[0], 3)  # If syndrome = 11, correct qubit 0
    qc.x(1).c_if(qc.cregs[0], 1)  # If syndrome = 01, correct qubit 1
    qc.x(2).c_if(qc.cregs[0], 2)  # If syndrome = 10, correct qubit 2

# Create a quantum circuit with 3 data qubits + 2 classical bits for syndrome
qc = QuantumCircuit(3, 2)

# Encode logical qubit
encode(qc)

# Introduce a random X error
introduce_error(qc)

# Measure syndrome before correction
measure_syndrome(qc)

# Apply error correction
correct_error(qc)

# Measure final state of qubit 0 to verify correction
qc.measure(0, 0)

# Run simulation using AerSimulator.run() (modern alternative to execute)
simulator = AerSimulator()
result = simulator.run(qc, shots=1000).result()

# Get results
counts = result.get_counts()
print("Final measurement results:", counts)
plot_histogram(counts)
