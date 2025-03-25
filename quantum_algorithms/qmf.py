import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

#  Quantum Minimum Finding
def quantum_minimum_finding(qc, qubits):
    n = len(qubits)
    if n <= 1:
        return
    
    # Apply Hadamard
    for qubit in qubits:
        qc.h(qubit)
    
    #  Grover-like amplification 
    for qubit in qubits:
        qc.x(qubit)
        qc.h(qubit)
        qc.x(qubit)
    
    #  classical extraction
    qc.measure_all()

# sorting using QMF
def quantum_sort(numbers):
    sorted_list = []
    remaining_numbers = numbers[:]
    
    while remaining_numbers:
        qc = QuantumCircuit(len(remaining_numbers), len(remaining_numbers))
        qubits = list(range(len(remaining_numbers)))
        quantum_minimum_finding(qc, qubits)
        
        simulator = AerSimulator()
        result = simulator.run(qc).result()
        counts = result.get_counts()
        
        # Extract minimum index
        min_index = min(counts, key=counts.get)
        sorted_list.append(remaining_numbers[int(min_index, 2)])
        remaining_numbers.pop(int(min_index, 2))
    
    return sorted_list

numbers = np.random.randint(1, 100, size=4).tolist()
print("Unsorted list:", numbers)
sorted_numbers = quantum_sort(numbers)
print("Sorted list:", sorted_numbers)