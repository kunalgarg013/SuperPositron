from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import Shor
from qiskit.utils import algorithm_globals

# Set seed for reproducibility
algorithm_globals.random_seed = 42

# Define the integer to factor
N = 15

# Create the simulator backend
simulator = AerSimulator()

# Initialize and run the Shor's algorithm instance
shor_instance = Shor(simulator)
result = shor_instance.factor(N)

# Print the result
print("Factors of", N, ":", result.factors)
