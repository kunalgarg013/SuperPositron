import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset, random_split
from mlxtend.data import loadlocal_mnist
import sys

# Qiskit imports for version 2.0.2
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator

# Custom Quantum Layer Implementation
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, n_layers=3, n_outputs=6):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        
        # Initialize quantum weights as trainable parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers * n_qubits) * 0.1
        )
        
        # Create the quantum circuit template
        self.circuit_template = self._create_circuit_template()
        
        # Create observables (Pauli-Z measurements)
        self.observables = self._create_observables()
        
        # Initialize estimator
        self.estimator = Estimator()

    def _create_circuit_template(self):
        """Create parameterized quantum circuit"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Input parameters
        input_params = [Parameter(f'input_{i}') for i in range(self.n_qubits)]
        
        # Weight parameters
        weight_params = [Parameter(f'weight_{i}') for i in range(self.n_layers * self.n_qubits)]
        
        # Input encoding layer
        for i in range(self.n_qubits):
            qc.ry(input_params[i], i)
        
        # Entangling layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Parameterized rotations
            for qubit in range(self.n_qubits):
                qc.ry(weight_params[param_idx], qubit)
                param_idx += 1
            
            # Entangling gates
            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)
            # Circular entanglement
            if self.n_qubits > 1:
                qc.cx(self.n_qubits - 1, 0)
        
        return qc, input_params, weight_params

    def _create_observables(self):
        """Create Pauli-Z observables for each output"""
        observables = []
        for i in range(min(self.n_outputs, self.n_qubits)):
            pauli_string = ['I'] * self.n_qubits
            pauli_string[i] = 'Z'
            observable = SparsePauliOp.from_list([(''.join(pauli_string), 1.0)])
            observables.append(observable)
        return observables

    def forward(self, x):
        """Forward pass through quantum layer"""
        batch_size = x.shape[0]
        results = []
        
        qc_template, input_params, weight_params = self.circuit_template
        
        for i in range(batch_size):
            # Bind parameters
            param_dict = {}
            
            # Bind input parameters
            for j, param in enumerate(input_params):
                param_dict[param] = float(x[i, j])
            
            # Bind weight parameters
            for j, param in enumerate(weight_params):
                param_dict[param] = float(self.quantum_weights[j])
            
            # Create bound circuit
            bound_circuit = qc_template.assign_parameters(param_dict)
            
            # Compute expectation values
            expectation_values = []
            try:
                # Use statevector simulation for faster computation
                statevector = Statevector.from_instruction(bound_circuit)
                
                for obs in self.observables:
                    # exp_val = statevector.expectation_value(obs).real
                    exp_val = self.estimator.run(bound_circuit, observables=obs).result().values[0]
                    expectation_values.append(exp_val)
                
                results.append(expectation_values)
                
            except Exception as e:
                print(f"Quantum computation error: {e}")
                # Fallback to random values if quantum computation fails
                expectation_values = [0.0] * self.n_outputs
                results.append(expectation_values)
        
        return torch.tensor(results, dtype=torch.float32, requires_grad=True)

# Load MNIST dataset
print("Loading MNIST dataset...")
img, label = loadlocal_mnist(
    images_path='./dataset/MNIST/train-images-idx3-ubyte',
    labels_path='./dataset/MNIST/train-labels-idx1-ubyte'
)
img = img / 255.0
img = img.reshape(-1, 1, 28, 28)
img_tensor = torch.tensor(img, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long)

# Limit dataset for faster training
img_tensor = img_tensor[:500]
label_tensor = label_tensor[:500]

# Prepare dataset
dataset = TensorDataset(img_tensor, label_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch size
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Quantum circuit parameters
n_qubits = 6
n_outputs = 6

# Hybrid model
class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # CNN feature extractor
        self.cnn = resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, n_qubits)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        
        # Quantum layer
        self.qnn = QuantumLayer(n_qubits=n_qubits, n_layers=3, n_outputs=n_outputs)
        
        # Classical head
        self.head = nn.Sequential(
            nn.Linear(n_outputs, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Extract features with CNN
        x = self.cnn(x)
        x = self.dropout(x)
        
        # Normalize inputs for quantum layer (gates expect values in reasonable range)
        x = torch.tanh(x) * np.pi
        
        # Process through quantum layer
        q_out = self.qnn(x)
        
        # Final classification
        return self.head(q_out)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Note: Quantum layer computation happens on CPU, CNN on specified device
model = HybridModel(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

n_epochs = 30
train_losses, test_accuracies = [], []

print("Starting training...")
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Move images to CPU for quantum computation, then back to device
            outputs = model(images)
            outputs = outputs.to(device)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / max(batch_count, 1)
    train_losses.append(avg_loss)
    scheduler.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    eval_loss = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            try:
                outputs = model(images)
                outputs = outputs.to(device)
                
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue
    
    acc = 100 * correct / max(total, 1)
    test_accuracies.append(acc)
    print(f"Epoch {epoch+1}/{n_epochs} Complete - Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%")
    print("-" * 50)

# Plot results
if train_losses and test_accuracies:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label="Test Accuracy", color='green', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves_mnist_qiskit_2.0.2_custom.png", dpi=300)
    plt.show()
else:
    print("No training data to plot.")

print("Training complete.")
sys.exit(0)