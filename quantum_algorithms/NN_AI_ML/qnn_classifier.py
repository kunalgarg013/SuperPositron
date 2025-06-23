# Modular QNN Classifier using Qiskit only as backend

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# ==== Hyperparameters ====
num_qubits = 4
num_layers = 2
shots = 128

# ==== Parameter Handler ====
class ParamSet:
    def __init__(self, num_layers, num_qubits):
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.vector = np.random.uniform(0, 2 * np.pi, num_layers * num_qubits * 2)

    def get_layer_params(self, layer):
        start = layer * self.num_qubits * 2
        end = start + self.num_qubits * 2
        return self.vector[start:end].reshape((self.num_qubits, 2))

    def update(self, new_vector):
        self.vector = new_vector

# ==== Circuit Generator ====
def build_quantum_circuit(input_vec, param_set):
    qc = QuantumCircuit(param_set.num_qubits, 1)

    for i, x_i in enumerate(input_vec):
        qc.ry(np.pi * x_i, i)

    for layer in range(param_set.num_layers):
        layer_weights = param_set.get_layer_params(layer)
        for i in range(param_set.num_qubits):
            rz, rx = layer_weights[i]
            qc.rz(rz, i)
            qc.rx(rx, i)
        for i in range(param_set.num_qubits):
            qc.cx(i, (i + 1) % param_set.num_qubits)

    qc.measure(param_set.num_qubits - 1, 0)
    return qc

# ==== Qiskit Backend Runner ====
def run_circuit(circuit, backend, shots=128):
    result = backend.run(circuit, shots=shots).result()
    qdist = result.quasi_dists[0]
    prob_1 = sum(
        p for bitstring, p in qdist.items()
        if format(bitstring, f"0{circuit.num_qubits}b")[-1] == '1'
    )
    return prob_1

# ==== Dataset Evaluator ====
def predict_dataset(X, param_set, backend, shots=128):
    return np.array([
        run_circuit(build_quantum_circuit(x, param_set), backend, shots)
        for x in tqdm(X, desc="Predicting", ncols=80)
    ])

# ==== Loss Function ====
def cost_fn(param_vector, param_set, X, y, backend):
    param_set.update(param_vector)
    preds = predict_dataset(X, param_set, backend)
    epsilon = 1e-10
    loss = -np.mean(y * np.log(preds + epsilon) + (1 - y) * np.log(1 - preds + epsilon))
    return loss

# ==== Dataset Generation ====
X_train = np.random.rand(50, num_qubits)
y_train = np.random.randint(0, 2, 50)
X_test = np.random.rand(20, num_qubits)
y_test = np.random.randint(0, 2, 20)

# ==== Training Loop ====
param_set = ParamSet(num_layers, num_qubits)
backend = Sampler()
init_weights = param_set.vector.copy()

opt_result = minimize(
    cost_fn, init_weights,
    args=(param_set, X_train, y_train, backend),
    method='COBYLA', options={'maxiter': 30}
)
param_set.update(opt_result.x)

# ==== Evaluation ====
y_pred_probs = predict_dataset(X_test, param_set, backend)
y_pred = (y_pred_probs > 0.5).astype(int)
quantum_acc = accuracy_score(y_test, y_pred)

# ==== Classical Baseline ====
clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
clf.fit(X_train, y_train)
y_classical = clf.predict(X_test)
classical_acc = accuracy_score(y_test, y_classical)

# ==== Plot Accuracy Comparison ====
plt.bar(["Quantum", "Classical"], [quantum_acc, classical_acc])
plt.ylim(0, 1)
plt.title("Classifier Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()