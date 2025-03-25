import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Define circuit
num_qubits = 8  
qc = QuantumCircuit(num_qubits)

# input parameters
input_params = ParameterVector("x", length=num_qubits)
theta = ParameterVector("\u03b8", length=num_qubits)

# add parameterized gates
for i in range(num_qubits):
    qc.ry(input_params[i], i)
    qc.rz(theta[i], i)

# qc.draw("mpl")
# plt.show()

# Quantum Neural Network
estimator = StatevectorEstimator()
qnn = EstimatorQNN(circuit=qc, input_params=input_params, weight_params=theta, estimator=estimator)

# Quantum Classifier
optimizer = COBYLA()
quantum_classifier = NeuralNetworkClassifier(neural_network=qnn, optimizer=optimizer)

# random data set
X_train = np.random.rand(2000, num_qubits)
y_train = np.random.randint(0, 2, 2000)

X_test = np.random.rand(500, num_qubits)
y_test = np.random.randint(0, 2, 500)

# train Quantum 
quantum_classifier.fit(X_train, y_train)
y_pred_quantum = quantum_classifier.predict(X_test)

# Classical MLP Classifier
classical_classifier = MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=1000)
classical_classifier.fit(X_train, y_train)
y_pred_classical = classical_classifier.predict(X_test)

# calculate accuracies
quantum_accuracy = accuracy_score(y_test, y_pred_quantum)
classical_accuracy = accuracy_score(y_test, y_pred_classical)

# Comparison
labels = ["Quantum Classifier", "Classical MLP"]
accuracies = [quantum_accuracy, classical_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(labels, accuracies, color=["blue", "green"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Quantum vs Classical Classifier Performance")
plt.show()

print(f"Quantum Accuracy: {quantum_accuracy:.2f}")
print(f"Classical Accuracy: {classical_accuracy:.2f}")