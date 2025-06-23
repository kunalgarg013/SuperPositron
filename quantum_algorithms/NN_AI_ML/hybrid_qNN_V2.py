import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset, random_split
from mlxtend.data import loadlocal_mnist
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler
import os
from tqdm import tqdm


# --- Quantum Layer (unchanged) ---
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

# --- Updated Hybrid Model (V2) ---
class HybridModelV2(nn.Module):
    def __init__(self, num_classes=10, n_qubits=6):
        super().__init__()
        self.n_qubits = n_qubits

        self.cnn = resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 64)
        self.dropout = nn.Dropout(0.5)

        self.latent_proj = nn.Linear(64, n_qubits)
        self.qnn = QuantumLayer(n_qubits=n_qubits)

        self.fc = nn.Sequential(
            nn.Linear(64 + n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        cnn_out = self.dropout(self.cnn(x))
        proj_input = self.latent_proj(cnn_out)
        qnn_out = self.qnn(proj_input)
        combined = torch.cat((cnn_out, qnn_out), dim=1)
        return self.fc(combined)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModelV2().to(device)
# 🔒 Load from your original trained model
load_checkpoint_path = "./analysis/checkpoint_epoch_latest.pt"

# 💾 Save progress to a *new* file (DO NOT overwrite original)
save_checkpoint_path = "./analysisV2/checkpoint_v2_finetune.pt"

# Initialize fresh optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
start_epoch = 0
train_losses, test_accs = [], []

# Load only model weights from checkpoint
if os.path.exists(load_checkpoint_path):
    checkpoint = torch.load(load_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ Loaded model weights from {load_checkpoint_path}")

    # Optional: reuse previous history
    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        test_accs = checkpoint['test_accs']
        start_epoch = checkpoint.get('epoch', 0)
else:
    print("⚠️ Warning: Original checkpoint file not found.")
    
    # Optionally load training history
    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        test_accs = checkpoint['test_accs']

# Load Data
data, labels = loadlocal_mnist(
    images_path='./dataset/MNIST/train-images-idx3-ubyte',
    labels_path='./dataset/MNIST/train-labels-idx1-ubyte'
)
data = data / 255.0
imgs = torch.tensor(data.reshape(-1, 1, 28, 28), dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(imgs[:1000], labels[:1000])
train_len = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Model, Loss, Optimizer
criterion = nn.CrossEntropyLoss()

# Try to resume checkpoint (from new V2 checkpoint)
if os.path.exists(save_checkpoint_path):
    checkpoint = torch.load(save_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    train_losses = checkpoint['train_losses']
    test_accs = checkpoint['test_accs']
    start_epoch = checkpoint['epoch']
    print(f"✅ Resumed training from epoch {start_epoch}")


# Training Loop
n_epochs = checkpoint.get('epoch', 0) + 5  # or whatever more you want

for epoch in range(start_epoch, n_epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)


    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Update progress bar with current loss
        loop.set_postfix(loss=loss.item())

    train_losses.append(total_loss / len(train_loader))

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    test_accs.append(acc)

    print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Accuracy={acc:.2f}%")

   # --------- Save to a new file to avoid overwriting ---------
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'test_accs': test_accs
    }, save_checkpoint_path)

    print(f"💾 Checkpoint saved at epoch {epoch + 1} to {save_checkpoint_path}")

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Loss", marker='o')
plt.title("Training Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accs, label="Accuracy", marker='o', color='green')
plt.title("Test Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("hybrid_qnn_v2_results.png")
plt.show()
