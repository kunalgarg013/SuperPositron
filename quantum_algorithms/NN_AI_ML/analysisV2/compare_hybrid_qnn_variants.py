import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from mlxtend.data import loadlocal_mnist
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler
import time
import matplotlib.pyplot as plt
import types

# --- Setup ---
n_qubits = 6
checkpoint_path = "checkpoint_v2_finetune.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Test Data ---
img, label = loadlocal_mnist(
    images_path='/home/kunal.g@qpiai.tech/QAlgo/NN_AI_ML/dataset/MNIST/t10k-images-idx3-ubyte',
    labels_path='/home/kunal.g@qpiai.tech/QAlgo/NN_AI_ML/dataset/MNIST/t10k-labels-idx1-ubyte'
)
img = img / 255.0
img = img.reshape(-1, 1, 28, 28)
img_tensor = torch.tensor(img, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long)

test_dataset = torch.utils.data.TensorDataset(img_tensor, label_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Quantum Layer ---
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=6, shots=256):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.data_params = ParameterVector('x', n_qubits)
        self.weight_params = ParameterVector('w', n_qubits)
        self.qc = self._build_template()
        self.weights = nn.Parameter(torch.rand(n_qubits) * 2 * np.pi)
        self.sampler = Sampler()

    def _build_template(self):
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(self.data_params[i], i)
            qc.rx(self.weight_params[i], i)
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def forward(self, x):
        circuits = []
        w = self.weights.detach().cpu().numpy()
        for sample in x:
            input_angles = (sample[:self.n_qubits] * 2 * np.pi).detach().cpu().numpy()
            binding = dict(zip(self.data_params, input_angles)) | dict(zip(self.weight_params, w))
            circuits.append(self.qc.assign_parameters(binding))

        job = self.sampler.run(circuits, shots=self.shots)
        results = job.result()
        features = []

        for dist in results.quasi_dists:
            probs = np.zeros(self.n_qubits)
            for bitstring, p in dist.items():
                bits = format(bitstring, f'0{self.n_qubits}b')
                for i in range(self.n_qubits):
                    probs[i] += p * int(bits[i])
            features.append(probs)

        return torch.tensor(features, dtype=torch.float32, device=x.device)

# --- Hybrid Model ---
class HybridModel(nn.Module):
    def __init__(self, use_cnn=True, use_qnn=True, num_classes=10):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_qnn = use_qnn

        if use_cnn:
            self.cnn = resnet18(weights=None)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 64)
            self.dropout = nn.Dropout(0.5)
            if use_qnn:
                self.latent_proj = nn.Linear(64, n_qubits)

        if use_qnn:
            self.qnn = QuantumLayer(n_qubits=n_qubits)

        # Combined features: CNN (64) + QNN (n_qubits)
        in_features = (64 if use_cnn else 0) + (n_qubits if use_qnn else 0)
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        parts = []
        if self.use_cnn:
            cnn_out = self.dropout(self.cnn(x))
            parts.append(cnn_out)
            if self.use_qnn:
                proj_input = self.latent_proj(cnn_out)
                qnn_out = self.qnn(proj_input)
                parts.append(qnn_out)
        elif self.use_qnn:
            x = x.view(x.size(0), -1)
            qnn_out = self.qnn(x[:, :n_qubits])
            parts.append(qnn_out)

        combined = torch.cat(parts, dim=1)
        return self.fc(combined)


# --- Evaluation Function ---
def evaluate(model, name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"📊 {name}: Accuracy = {acc:.2f}%")
    return acc

# --- Load Full Model Weights ---
full_model = HybridModel(use_cnn=True, use_qnn=True).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
full_model.load_state_dict(checkpoint['model_state_dict'])

# --- Clone Variants ---
# CNN-only model
cnn_only = HybridModel(use_cnn=True, use_qnn=False).to(device)
cnn_only.fc = nn.Sequential(
    nn.Linear(70, 64),  # Match checkpoint dimensions
    nn.ReLU(),
    nn.Linear(64, 10)
)
def cnn_forward(self, x):
    cnn_out = self.dropout(self.cnn(x))
    # Pad output to match original model dimensions
    padding = torch.zeros(cnn_out.shape[0], 70 - 64, device=cnn_out.device)
    cnn_out = torch.cat([cnn_out, padding], dim=1)
    return self.fc(cnn_out)
cnn_only.forward = types.MethodType(cnn_forward, cnn_only)

# QNN-only model 
qnn_only = HybridModel(use_cnn=False, use_qnn=True).to(device)
qnn_only.fc = nn.Sequential(
    nn.Linear(70, 64),  # Match checkpoint dimensions
    nn.ReLU(),
    nn.Linear(64, 10)
)
def qnn_forward(self, x):
    x = x.view(x.size(0), -1)
    q_out = self.qnn(x[:, :n_qubits])
    # Pad output to match original model dimensions
    padding = torch.zeros(q_out.shape[0], 70 - n_qubits, device=q_out.device)
    q_out = torch.cat([q_out, padding], dim=1)
    return self.fc(q_out)
qnn_only.forward = types.MethodType(qnn_forward, qnn_only)

# Load weights with strict=False to ignore mismatched layers
_ = cnn_only.load_state_dict(checkpoint['model_state_dict'], strict=False)
_ = qnn_only.load_state_dict(checkpoint['model_state_dict'], strict=False)



# --- Run Evaluations ---
evaluate(full_model, "Full Hybrid")
evaluate(cnn_only, "CNN Only")
evaluate(qnn_only, "QNN Only")


# --- Evaluation with Timing ---
def evaluate(model, name):
    model.eval()
    correct, total = 0, 0
    start = time.time()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    elapsed = time.time() - start
    acc = 100 * correct / total
    print(f"📊 {name}: Accuracy = {acc:.2f}%, Inference Time = {elapsed:.2f} seconds")
    return acc, elapsed

# --- Evaluate All ---
accs = {}
times = {}

# Fix the model dictionary and evaluation loop
models = {
    'Full Hybrid': full_model,
    'CNN Only': cnn_only,
    'QNN Only': qnn_only
}

for name, model in models.items():
    acc, elapsed = evaluate(model, name)
    accs[name] = acc
    times[name] = elapsed

# --- Plot Accuracy Comparison ---
plt.figure(figsize=(10, 4))
plt.bar(accs.keys(), accs.values(), color=['blue', 'gray', 'purple'])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("qnn_model_accuracy_comparison.png")
plt.show()

# --- Print Timing Summary ---
print("\n⏱ Inference Time (seconds):")
for name, time in times.items():
    print(f"• {name}: {time:.2f}s")

