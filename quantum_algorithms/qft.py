from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def qft(n):
    """Creates a Quantum Fourier Transform circuit for n qubits."""
    qc = QuantumCircuit(n)

    qc.s(0)  
    qc.t(2)  

    
    for i in range(n):
        qc.h(i)  # Apply Hadamard to the i-th qubit
        for j in range(i + 1, n):
            qc.cp(2 * 3.14159 / (2 ** (j - i + 1)), j, i)  # Controlled phase shift

    # Reverse the order of qubits 
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    return qc

# Create a 3-qubit QFT circuit
n = 4
qft_circuit = qft(n)

qft_circuit.measure_all()
print(qft_circuit.draw("mpl"))

simulator = AerSimulator() 
tqc = transpile(qft_circuit, simulator)
result = simulator.run(tqc, shots=1024).result()

counts = result.get_counts()
plot_histogram(counts)
plt.show()
