from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA

# -----------------------
# Build Hamiltonian
# -----------------------

def qubit_var(i, n_qubits):
    z = ['0'] * n_qubits
    z[i] = '1'
    z_mask = ''.join(z)

    # Create Pauli operators in symplectic format: (z, x)
    # Here, x = '0' * n_qubits, since we're using only Z
    pauli_z = Pauli((z_mask, '0' * n_qubits))
    identity = Pauli(('0' * n_qubits, '0' * n_qubits))

    return SparsePauliOp.from_list([
        (pauli_z, -0.5),
        (identity, 0.5)
    ])

n = 4  # 2 qubits for x, 2 for y
x0, x1, y0, y1 = 0, 1, 2, 3

x_op = qubit_var(x0, n) + 2 * qubit_var(x1, n)
y_op = qubit_var(y0, n) + 2 * qubit_var(y1, n)

N = 15
hamiltonian = (x_op @ y_op - N) @ (x_op @ y_op - N)

# -----------------------
# VQE Setup
# -----------------------

ansatz = TwoLocal(n, 'ry', 'cz', reps=2, entanglement='full')
optimizer = COBYLA(maxiter=200)
vqe = VQE(ansatz=ansatz, optimizer=optimizer)

result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

print("Minimum energy:", result.eigenvalue.real)
print("Optimal parameters:", result.optimal_parameters)

# -----------------------
# Postprocessing: Final State
# -----------------------

# Bind optimal parameters
bound_circuit = ansatz.bind_parameters(result.optimal_parameters)
state = Statevector.from_instruction(bound_circuit)
counts = state.sample_counts(shots=1024)

# Most likely result
top_state = max(counts.items(), key=lambda x: x[1])[0]
print(f"\nMost likely bitstring: {top_state}")

# Extract x and y from bitstring (q0 = rightmost)
x_bits = top_state[-1:-3:-1]  # qubit 0 and 1
y_bits = top_state[-3:-5:-1]  # qubit 2 and 3

x = int(x_bits, 2)
y = int(y_bits, 2)
print(f"Recovered factors: x = {x}, y = {y}  → x*y = {x * y}")
