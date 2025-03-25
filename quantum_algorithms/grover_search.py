from qiskit import QuantumCircuit, transpile  
from qiskit_aer import Aer  
from qiskit_aer.backends import AerSimulator  

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def grover_oracle(qc, target_state): # Applies the oracle for marking |101⟩.
    qc.x([0, 2])  # Flip qubits
    qc.h(2)  # Control on qubit 2
    qc.ccx(0, 1, 2)  # Toffoli gate
    qc.h(2)
    qc.x([0, 2])  # Unflip 

def diffusion_operator(qc): #  diffusion operator.
    qc.h([0, 1, 2])  #  Hadamard to all
    qc.x([0, 1, 2])  # Flip qubits
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])  # Unflip
    qc.h([0, 1, 2])  # Final Hadamard


qc = QuantumCircuit(3, 3)
qc.h([0, 1, 2])
grover_oracle(qc, "101")
diffusion_operator(qc)
qc.measure([0, 1, 2], [0, 1, 2])

# Simulate the circuit
simulator = AerSimulator()
tqc = transpile(qc, simulator)
result = simulator.run(tqc, shots=10000000).result()
counts = result.get_counts()

qc.draw('mpl')
plot_histogram(counts)
plt.show()
