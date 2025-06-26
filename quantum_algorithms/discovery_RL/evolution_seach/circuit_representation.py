import random
import numpy as np
import copy
from qiskit import QuantumCircuit

# Expanded gate sets
SINGLE_QUBIT_PARAM_GATES = ['rx', 'ry', 'rz']
SINGLE_QUBIT_FIXED_GATES = ['h', 'x', 'y', 's', 't', 'sx', 'id']
TWO_QUBIT_GATES = ['cx', 'cz', 'swap']
THREE_QUBIT_GATES = ['ccx', 'cswap']
ALL_GATES = SINGLE_QUBIT_PARAM_GATES + SINGLE_QUBIT_FIXED_GATES + TWO_QUBIT_GATES + THREE_QUBIT_GATES

class QuantumCircuitCandidate:
    def __init__(self, num_qubits, max_depth, gate_sequence=None):
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.gate_sequence = gate_sequence or []

    def random_initialize(self, max_gates=None):
        max_gates = max_gates or self.max_depth * self.num_qubits
        self.gate_sequence = []
        for _ in range(random.randint(1, max_gates)):
            self.add_gate()

    def add_gate(self):
        gate = random.choice(ALL_GATES)

        if gate in SINGLE_QUBIT_PARAM_GATES:
            qubit = random.randint(0, self.num_qubits - 1)
            angle = random.uniform(0, 2 * np.pi)
            self.gate_sequence.append({'gate': gate, 'qubit': qubit, 'angle': angle})

        elif gate in SINGLE_QUBIT_FIXED_GATES:
            qubit = random.randint(0, self.num_qubits - 1)
            self.gate_sequence.append({'gate': gate, 'qubit': qubit})

        elif gate in TWO_QUBIT_GATES:
            if self.num_qubits < 2:
                return
            q1, q2 = random.sample(range(self.num_qubits), 2)
            self.gate_sequence.append({'gate': gate, 'qubits': [q1, q2]})

        elif gate in THREE_QUBIT_GATES:
            if self.num_qubits < 3:
                return
            q1, q2, q3 = random.sample(range(self.num_qubits), 3)
            self.gate_sequence.append({'gate': gate, 'qubits': [q1, q2, q3]})

    def to_qiskit_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        for g in self.gate_sequence:
            gate = g['gate']
            if gate in SINGLE_QUBIT_PARAM_GATES:
                getattr(qc, gate)(g['angle'], g['qubit'])
            elif gate in SINGLE_QUBIT_FIXED_GATES:
                getattr(qc, gate)(g['qubit'])
            elif gate in TWO_QUBIT_GATES:
                qc.__getattribute__(gate)(*g['qubits'])
            elif gate == 'ccx':
                qc.ccx(*g['qubits'])
            elif gate == 'cswap':
                qc.cswap(*g['qubits'])
        return qc

    def copy(self):
        return QuantumCircuitCandidate(
            self.num_qubits,
            self.max_depth,
            gate_sequence=copy.deepcopy(self.gate_sequence)
        )

    def mutate_gate(self):
        if not self.gate_sequence:
            return
        idx = random.randint(0, len(self.gate_sequence) - 1)
        gate_info = self.gate_sequence[idx]
        gate = gate_info['gate']

        if gate in SINGLE_QUBIT_PARAM_GATES:
            if random.random() < 0.5:
                gate_info['angle'] += random.uniform(-0.5, 0.5)
            else:
                gate_info['qubit'] = random.randint(0, self.num_qubits - 1)
        elif gate in SINGLE_QUBIT_FIXED_GATES:
            gate_info['qubit'] = random.randint(0, self.num_qubits - 1)
        elif gate in TWO_QUBIT_GATES:
            q1, q2 = random.sample(range(self.num_qubits), 2)
            gate_info['qubits'] = [q1, q2]
        elif gate in THREE_QUBIT_GATES:
            q1, q2, q3 = random.sample(range(self.num_qubits), 3)
            gate_info['qubits'] = [q1, q2, q3]

    def mutate(self, mutation_rate=0.3):
        if random.random() < mutation_rate:
            self.mutate_gate()
        if random.random() < mutation_rate:
            self.add_gate()
        if random.random() < mutation_rate:
            self.remove_gate()

    def remove_gate(self):
        if self.gate_sequence:
            idx = random.randint(0, len(self.gate_sequence) - 1)
            del self.gate_sequence[idx]

    def describe(self):
        print(f"QuantumCircuitCandidate with {len(self.gate_sequence)} gates")
        for g in self.gate_sequence:
            print(g)

    def gate_count(self):
        return len(self.gate_sequence)

    def depth_estimate(self):
        return min(len(self.gate_sequence), self.max_depth)
