import random
from copy import deepcopy
from qiskit import QuantumCircuit
from circuit_representation import QuantumCircuitCandidate
from fitness_evaluation import compute_fitness
from parameter_optimizer import optimize_parameters
from known_states import get_ghz_statevector

def crossover(parent1, parent2):
    gates1 = parent1.gate_sequence
    gates2 = parent2.gate_sequence

    if not gates1 or not gates2:
        return parent1.copy()

    cut1 = random.randint(1, len(gates1))
    cut2 = random.randint(1, len(gates2))

    child_sequence = gates1[:cut1] + gates2[cut2:]
    child = QuantumCircuitCandidate(parent1.num_qubits, parent1.max_depth)
    child.gate_sequence = deepcopy(child_sequence)
    return child

def ghz_circuit_gate_sequence(num_qubits):
    """Returns gate sequence for GHZ state"""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    sequence = []
    for instr, qargs, _ in qc.data:
        if instr.name == 'h':
            sequence.append({'gate': 'h', 'qubit': qargs[0]._index})
        elif instr.name in ['cx', 'cz']:
            sequence.append({'gate': instr.name, 'qubits': [qargs[0]._index, qargs[1]._index]})
    return sequence

    """Returns gate sequence for GHZ state"""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    sequence = []
    for instr, qargs, _ in qc.data:
        if instr.name == 'h':
            sequence.append({'gate': 'h', 'qubit': qargs[0]._index})
        elif instr.name == 'cx':
            sequence.append({'gate': 'cx', 'qubits': [qargs[0]._index, qargs[1]._index]})
    return sequence

class Population:
    def __init__(self, size, num_qubits, max_depth, alpha=0.05, beta=0.01):
        self.size = size
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.alpha = alpha
        self.beta = beta
        self.individuals = []

    def initialize(self):
        self.individuals = []

        # Inject one known GHZ circuit
        seed = QuantumCircuitCandidate(self.num_qubits, self.max_depth)
        seed.gate_sequence = ghz_circuit_gate_sequence(self.num_qubits)
        self.individuals.append(seed)
        
        for _ in range(self.size):
            ind = QuantumCircuitCandidate(self.num_qubits, self.max_depth)
            ind.random_initialize()
            self.individuals.append(ind)

    def evaluate(self, target_state):
        scores = []
        for ind in self.individuals:
            fitness = compute_fitness(ind, target_state, self.alpha, self.beta)
            scores.append((fitness, ind))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores

    def evolve(self, target_state, retain_top_k=5, mutate_rate=0.3, crossover_rate=0.4, optimize_top_k=True):
        """
        Evolve the population using mutation, crossover, and optional parameter optimization.
        """
        scored = self.evaluate(target_state)
        top_individuals = [ind.copy() for _, ind in scored[:retain_top_k]]

        if optimize_top_k:
            for i in range(len(top_individuals)):
                top_individuals[i] = optimize_parameters(
                    top_individuals[i], target_state,
                    alpha=self.alpha, beta=self.beta,
                    steps=50, lr=0.1
                )

        new_generation = []
        while len(new_generation) < self.size:
            if random.random() < crossover_rate and len(top_individuals) > 1:
                p1, p2 = random.sample(top_individuals, 2)
                child = crossover(p1, p2)
            else:
                child = random.choice(top_individuals).copy()
                child.mutate(mutation_rate=mutate_rate)
            new_generation.append(child)

        self.individuals = new_generation
        return self.evaluate(target_state)[0][0]  # Return best fitness after optimization
