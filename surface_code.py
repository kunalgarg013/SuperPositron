import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, visualize_transition
import matplotlib.pyplot as plt

# Function to create a simple surface code layout
def create_surface_code(qc):
    data_qubits = [0, 1, 2, 3]
    for q in data_qubits:
        qc.h(q)
    
    qc.cx(0, 4)
    qc.cx(1, 4)
    qc.cx(1, 5)
    qc.cx(2, 5)
    qc.cx(2, 6)
    qc.cx(3, 6)
    qc.cx(3, 7)
    qc.cx(0, 7)

# Function to introduce a fixed X or Z error on a specific qubit
def introduce_error(qc, error_qubit=0, error_type='X'):
    if error_type == 'X':
        qc.x(error_qubit)
    else:
        qc.z(error_qubit)
    print(f"Introduced {error_type} error on qubit {error_qubit}")

# Function to measure the stabilizers (syndrome extraction)
def measure_syndrome(qc, syndrome):
    qc.measure([4, 5, 6, 7], syndrome)  # Measure stabilizer qubits

# Function to correct errors based on syndrome measurements
def correct_errors(qc, syndrome):
    qc.barrier()
    error_map = {
        '0001': 0, '0010': 1, '0100': 2, '1000': 3
    }
    
    for syndrome_bits, qubit in error_map.items():
        with qc.if_test((syndrome, int(syndrome_bits, 2))):
            qc.x(qubit)
            print(f"Correction applied to qubit {qubit}")

# Create a quantum circuit with 4 data qubits, 4 stabilizer qubits, and 4 classical bits for syndrome
qubits = QuantumRegister(8, "q")
syndrome = ClassicalRegister(4, "c")
qc_no_correction = QuantumCircuit(qubits, syndrome)
qc_corrected = QuantumCircuit(qubits, syndrome)

# Set up the surface code
create_surface_code(qc_no_correction)
create_surface_code(qc_corrected)

# Introduce the same error in both circuits
error_qubit = 0  # Fixed for consistent results
error_type = 'X'  # Choose between 'X' and 'Z'
introduce_error(qc_no_correction, error_qubit, error_type)
introduce_error(qc_corrected, error_qubit, error_type)

# Measure syndrome
measure_syndrome(qc_no_correction, syndrome)
measure_syndrome(qc_corrected, syndrome)

# Apply error correction only in the second circuit
correct_errors(qc_corrected, syndrome)

# Run simulation
simulator = AerSimulator()
result_no_correction = simulator.run(qc_no_correction, shots=1000).result()
result_corrected = simulator.run(qc_corrected, shots=1000).result()

counts_no_correction = result_no_correction.get_counts()
counts_corrected = result_corrected.get_counts()

# Display results
print("Syndrome measurement results before correction:", counts_no_correction)
print("Syndrome measurement results after correction:", counts_corrected)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_histogram(counts_no_correction, ax=axes[0])
axes[0].set_title("Before Error Correction")
plot_histogram(counts_corrected, ax=axes[1])
axes[1].set_title("After Error Correction")
plt.show()

# Bloch Sphere Visualization
print("Visualizing Bloch Sphere for a single qubit")
visualize_transition((0, 0), (1, 0))
