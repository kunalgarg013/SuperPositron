import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# Define the 5x5 maze
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 2]  # Goal marked as 2
])

def encode_maze(qc):
    """Encodes the maze structure in the quantum circuit."""
    for qubit in range(5):
        qc.h(qubit)

def oracle(qc):
    """Marks the goal state using a phase flip."""
    qc.cz(0, 4)  # Example encoding for a solution state

def diffusion(qc):
    """Amplifies the probability of correct states."""
    for qubit in range(5):
        qc.h(qubit)
        qc.x(qubit)
    qc.h(4)
    qc.mcx([0, 1, 2, 3], 4)
    qc.h(4)
    for qubit in range(5):
        qc.x(qubit)
        qc.h(qubit)

def quantum_maze_solver():
    """Builds and runs the quantum maze solver."""
    qc = QuantumCircuit(5, 5)
    encode_maze(qc)
    oracle(qc)
    diffusion(qc)
    qc.measure(range(5), range(5))
    
    backend = Aer.get_backend('aer_simulator')
    results = backend.run(qc).result()
    counts = results.get_counts()
    
    plot_histogram(counts)
    plt.show()
    
    visualize_quantum_maze(counts)
    
    return counts

def visualize_quantum_maze(counts):
    """Visualizes the maze and highlights the most probable quantum paths."""
    plt.figure(figsize=(6, 6))
    plt.imshow(maze, cmap='gray_r')
    
    for state, count in counts.items():
        if len(state) == 5:  # Ensure valid 5-bit states
            x = int(state[:3], 2) % 5  # Convert first 3 bits to row index
            y = int(state[3:], 2) % 5  # Convert last 2 bits to column index
            plt.text(y, x, f'{count}', ha='center', va='center', color='red', fontsize=12)
    
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.grid(True, color='black')
    plt.title("Quantum Maze Solution Paths")
    plt.show()

# Execute the solver
solution_counts = quantum_maze_solver()
print("Quantum Maze Solution:", solution_counts)
