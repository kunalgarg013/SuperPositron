import random
from circuit_representation import QuantumCircuitCandidate
from fitness_evaluation import compute_fitness
from parameter_optimizer import optimize_parameters

class Population:
    def __init__(self, size, num_qubits, max_depth, alpha=0.05, beta=0.01):
        self.size = size
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.alpha = alpha
        self.beta = beta
        self.individuals = []

    def initialize(self, seed_gate_sequence=None):
        self.individuals = []
        for _ in range(self.size):
            circuit = QuantumCircuitCandidate(self.num_qubits, self.max_depth)
            if seed_gate_sequence:
                circuit.gate_sequence = seed_gate_sequence.copy()
            else:
                circuit.random_initialize()
            self.individuals.append(circuit)

    def evaluate(self, target_state, gamma=0.0):
        scores = []
        for ind in self.individuals:
            fitness = compute_fitness(ind, target_state, self.alpha, self.beta, gamma=gamma)
            scores.append((fitness, ind))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores

    def crossover(self, parent1, parent2):
        child = QuantumCircuitCandidate(self.num_qubits, self.max_depth)
        p1_seq = parent1.gate_sequence
        p2_seq = parent2.gate_sequence
        min_len = min(len(p1_seq), len(p2_seq))
        if min_len < 2:
            return random.choice([parent1, parent2]).copy()
        split = random.randint(1, min_len - 1)
        child.gate_sequence = p1_seq[:split] + p2_seq[split:]
        return child

    def evolve(self, target_state, retain_top_k=5, mutate_rate=0.3, crossover_rate=0.4, optimize_top_k=True, gamma=0.0):
        scored = self.evaluate(target_state, gamma=gamma)
        top_k = [ind.copy() for _, ind in scored[:retain_top_k]]

        # Optional optimization of elite individuals
        if optimize_top_k:
            for ind in top_k:
                optimize_parameters(ind, target_state, steps=200, lr=0.01)

        # Generate offspring
        offspring = []
        while len(offspring) < self.size - retain_top_k:
            if random.random() < crossover_rate and len(top_k) >= 2:
                parent1 = random.choice(top_k)
                parent2 = random.choice(top_k)
                child = self.crossover(parent1, parent2)
            else:
                child = random.choice(top_k).copy()
            child.mutate(mutation_rate=mutate_rate)
            offspring.append(child)

        # Inject a few fresh random candidates
        n_fresh = max(1, self.size // 10)
        fresh = []
        for _ in range(n_fresh):
            c = QuantumCircuitCandidate(self.num_qubits, self.max_depth)
            c.random_initialize()
            fresh.append(c)

        # Form next generation
        self.individuals = top_k + offspring[:self.size - retain_top_k - n_fresh] + fresh

        return self.evaluate(target_state, gamma=gamma)[0][0]
