import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Quantum Bitonic Sort
def quantum_bitonic_sort(qc, qubits):
    n = len(qubits)
    if n <= 1:
        return
    
    for qubit in qubits:
        qc.h(qubit)
    
    for i in range(n):
        for j in range(i + 1, n):
            qc.cswap(i, j, (i + j) % n)

    qc.measure_all()

# Quantum sorting using bitonic
def quantum_sort(numbers):
    sorted_list = []
    remaining_numbers = numbers[:]
    
    while remaining_numbers:
        qc = QuantumCircuit(len(remaining_numbers), len(remaining_numbers))
        qubits = list(range(len(remaining_numbers)))
        quantum_bitonic_sort(qc, qubits)
        
        simulator = AerSimulator()
        result = simulator.run(qc).result()
        counts = result.get_counts()
        
        if not counts:
            print("No valid measurement results.")
            break
        
        # Extract 
        sorted_indices = sorted(counts, key=counts.get, reverse=True)
        try:
            sorted_ints = [int(idx.replace(" ", ""), 2) for idx in sorted_indices]
            sorted_list.extend([remaining_numbers[i] for i in sorted_ints if i < len(remaining_numbers)])
            break 
        except ValueError:
            print(f"Invalid binary index detected: {sorted_indices}")
            break
    
    return sorted_list


numbers = np.random.randint(1, 100, size=4).tolist()
print("Unsorted list:", numbers)
sorted_numbers = quantum_sort(numbers)
print("Sorted list:", sorted_numbers)