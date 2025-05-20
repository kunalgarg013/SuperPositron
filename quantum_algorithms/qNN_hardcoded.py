import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm


# Settings
num_qubits = 4
shots = 128

# Backend
sampler = Sampler()

# Data generation
X_train = np.random.rand(50, num_qubits)
y_train = np.random.randint(0, 2, 50)

X_test = np.random.rand(20, num_qubits)
y_test = np.random.randint(0, 2, 20)


# Define the quantum circuit template
def construct_circuit(x, weights):
    # qc = QuantumCircuit(num_qubits)
    qc = QuantumCircuit(num_qubits, 1)
    for i in range(num_qubits):
        qc.ry(np.pi * x[i], i)
        qc.rz(weights[i], i)
    # qc.measure_all()
    qc.measure(num_qubits - 1, 0)  # Measure only last qubit into classical bit 0
    return qc

# Define prediction function using expectation value
def predict_probs(weights, X):
    probs = []
    with tqdm(total=len(X), desc="Evaluating dataset", ncols=80) as pbar:
        for x in X:
            circuit = construct_circuit(x, weights)
            result = sampler.run(circuit, shots=shots).result()
            quasi_dist = result.quasi_dists[0]
            prob_one = sum(
                p for bitstring, p in quasi_dist.items()
                if format(bitstring, f"0{num_qubits}b")[-1] == '1'
            )
            probs.append(prob_one)
            pbar.update(1)
    return np.array(probs)



# Cost function (binary cross-entropy)
def cost_fn(weights):
    preds = predict_probs(weights, X_train)
    epsilon = 1e-10
    loss = -np.mean(y_train * np.log(preds + epsilon) + (1 - y_train) * np.log(1 - preds + epsilon))
    return loss

# Initialize random weights
np.random.seed(42)
init_weights = 2 * np.pi * np.random.rand(num_qubits)

# Optimization
opt_result = minimize(cost_fn, init_weights, method='COBYLA', options={'maxiter': 100})

# Evaluate on test set
final_weights = opt_result.x
y_pred_probs = predict_probs(final_weights, X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Accuracy
quantum_acc = accuracy_score(y_test, y_pred)
print(f"Quantum classifier accuracy: {quantum_acc:.2f}")

# Compare to classical baseline
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)
clf.fit(X_train, y_train)
y_classical = clf.predict(X_test)
classical_acc = accuracy_score(y_test, y_classical)

# Plot
labels = ["Quantum", "Classical"]
accs = [quantum_acc, classical_acc]
plt.bar(labels, accs)
plt.ylim(0, 1)
plt.title("Classifier Accuracy Comparison")
plt.show()
