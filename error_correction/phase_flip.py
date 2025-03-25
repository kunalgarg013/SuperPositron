import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# encode logical qubit using 3-qubit phase-flip code
def encode(qc):
    qc.h(0)  #  Hadamard 
    qc.cx(0, 1)
    qc.cx(0, 2)

# introduce a Z error 
def introduce_error(qc):
    error_qubit = np.random.choice([0, 1, 2])
    qc.z(error_qubit)
    print(f"Introduced Z error on qubit {error_qubit}")

# measure syndrome for phase errors
def measure_syndrome(qc, syndrome):
    qc.h([0, 1, 2])  # convert to phase error
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h([0, 1, 2])  # convert back to computational basis
    qc.measure([1, 2], syndrome)  # store syndrome

# correct phase error based on syndrome measurement
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

# create a quantum circuit
qubits = QuantumRegister(3, "q")
syndrome = ClassicalRegister(2, "c")
qc = QuantumCircuit(qubits, syndrome)

encode(qc)
introduce_error(qc)
measure_syndrome(qc, syndrome)
correct_error(qc, syndrome)

simulator = AerSimulator()
result = simulator.run(qc, shots=1000).result()
counts = result.get_counts()
print("Syndrome measurement results:", counts)
plot_histogram(counts).show()