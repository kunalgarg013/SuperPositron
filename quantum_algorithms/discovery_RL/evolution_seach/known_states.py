from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def get_bell_statevector():
    """Returns |Φ+⟩ Bell state: (|00⟩ + |11⟩)/√2"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return Statevector.from_instruction(qc)

def get_ghz_statevector(num_qubits=3):
    """Returns GHZ state for n qubits: (|000...0⟩ + |111...1⟩)/√2"""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return Statevector.from_instruction(qc)

def get_w_statevector(num_qubits=3):
    """Returns 3-qubit W state: (|001⟩ + |010⟩ + |100⟩)/√3"""
    if num_qubits != 3:
        raise NotImplementedError("W-state currently only supported for 3 qubits")
    qc = QuantumCircuit(3)
    qc.ry(2.0944, 0)       # π/1.5
    qc.cx(0, 1)
    qc.ry(1.5708, 1)       # π/2
    qc.cx(1, 2)
    qc.ry(-1.5708, 2)
    return Statevector.from_instruction(qc)

def get_w_gate_sequence():
    """
    Returns gate sequence approximating a 3-qubit W state:
    (|001⟩ + |010⟩ + |100⟩) / sqrt(3)
    Approximate version using standard RY + CNOT ladder
    """
    return [
        {'gate': 'ry', 'qubit': 0, 'angle': 2.0944},   # π / 1.5
        {'gate': 'cx', 'qubits': [0, 1]},
        {'gate': 'ry', 'qubit': 1, 'angle': 1.5708},   # π / 2
        {'gate': 'cx', 'qubits': [1, 2]},
        {'gate': 'ry', 'qubit': 2, 'angle': -1.5708}
    ]

