from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import Shor
from qiskit.utils import algorithm_globals

# Set seed 
algorithm_globals.random_seed = 42
N = 15

simulator = AerSimulator()

# Initialize and run the Shor's algorithm instance
shor_instance = Shor(simulator)
result = shor_instance.factor(N)


print("Factors of", N, ":", result.factors)
