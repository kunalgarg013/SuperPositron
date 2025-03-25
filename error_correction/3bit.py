import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

#  encode logical qubit using 3-qubit repetition code
def encode(qc):
    qc.cx(0, 1)
    qc.cx(0, 2)

#  introduce an X error on a random qubit
def introduce_error(qc):
    error_qubit = np.random.choice([0, 1, 2])
    qc.x(error_qubit)
    print(f"Introduced X error on qubit {error_qubit}")

#  detect error using syndrome measurement
def measure_syndrome(qc):
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([1, 2], [0, 1]) 

# correct error based on syndrome measurement
def correct_error(qc):
    qc.x(0).c_if(qc.cregs[0], 3)  # If syndrome = 11, correct qubit 0
    qc.x(1).c_if(qc.cregs[0], 1)  # If syndrome = 01, correct qubit 1
    qc.x(2).c_if(qc.cregs[0], 2)  # If syndrome = 10, correct qubit 2

# create a quantum circuit 
qc = QuantumCircuit(3, 2)

encode(qc)
introduce_error(qc)
measure_syndrome(qc)
correct_error(qc)

# measure
qc.measure(0, 0)
simulator = AerSimulator()
result = simulator.run(qc, shots=1000).result()

counts = result.get_counts()
print("Final measurement results:", counts)
plot_histogram(counts)
