import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Function to encode logical qubit using 3-qubit phase-flip code
def encode(qc):
    qc.h(0)  # Apply Hadamard to logical qubit
    qc.cx(0, 1)
    qc.cx(0, 2)

# Function to introduce a Z error on a random qubit
def introduce_error(qc):
    error_qubit = np.random.choice([0, 1, 2])
    qc.z(error_qubit)
    print(f"Introduced Z error on qubit {error_qubit}")

# Function to measure syndrome for phase errors
def measure_syndrome(qc, syndrome):
    qc.h([0, 1, 2])  # Convert to phase error detection basis
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h([0, 1, 2])  # Convert back to computational basis
    qc.measure([1, 2], syndrome)  # Store syndrome in classical bits

# Function to correct phase error based on syndrome measurement
def correct_error(qc, syndrome):
    qc.barrier()
    with qc.if_test((syndrome, 3)):
        qc.z(0)  # If syndrome is 11, correct qubit 0
        print("Correction applied to qubit 0")
    with qc.if_test((syndrome, 1)):
        qc.z(1)  # If syndrome is 01, correct qubit 1
        print("Correction applied to qubit 1")
    with qc.if_test((syndrome, 2)):
        qc.z(2)  # If syndrome is 10, correct qubit 2
        print("Correction applied to qubit 2")

# Create a quantum circuit with 3 data qubits + 2 classical bits for syndrome
qubits = QuantumRegister(3, "q")
syndrome = ClassicalRegister(2, "c")
qc = QuantumCircuit(qubits, syndrome)

# Encode logical qubit
encode(qc)

# Introduce a random Z error
introduce_error(qc)

# Measure syndrome before correction
measure_syndrome(qc, syndrome)

# Apply error correction
correct_error(qc, syndrome)

# Run simulation to check syndrome detection
simulator = AerSimulator()
result = simulator.run(qc, shots=1000).result()
counts = result.get_counts()
print("Syndrome measurement results:", counts)
plot_histogram(counts).show()