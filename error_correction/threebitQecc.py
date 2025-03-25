from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt  

# Define a 3-qubit Quantum Register and 1-bit Classical Register
qreg = QuantumRegister(3, 'q')
creg = ClassicalRegister(1, 'c')
qc = QuantumCircuit(qreg, creg)

# encode a bit-flip error correction code 
qc.h(0)        #  superposition
qc.cx(0, 1)    # copy state to qubit 1
qc.cx(0, 2)    # copy state to qubit 2

# Introduce a flip bit error
qc.x(1)  

# error correction
qc.cx(1, 0)
qc.cx(2, 0)

qc.measure(0, 0)
simulator = AerSimulator()
result = simulator.run(qc).result()
counts = result.get_counts()

print(counts)
plot_histogram(counts)
plt.show() 
