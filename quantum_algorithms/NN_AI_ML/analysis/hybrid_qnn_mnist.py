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
import sys
import os

# Checkpoint config
checkpoint_path = "checkpoint_epoch_latest.pt"
start_epoch = 0
train_losses, test_accs = [], []

# Try to load checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_losses = checkpoint['train_losses']
    test_accs = checkpoint['test_accs']
    start_epoch = checkpoint['epoch']
    print(f"✅ Resumed from epoch {start_epoch}")

# Dataset loading
img, label = loadlocal_mnist(
    images_path='./dataset/MNIST/train-images-idx3-ubyte',
    labels_path='./dataset/MNIST/train-labels-idx1-ubyte'
)
img = img / 255.0
img = img.reshape(-1, 1, 28, 28)
img_tensor = torch.tensor(img, dtype=torch.float32)
label_tensor = torch.tensor(label, dtype=torch.long)

dataset = TensorDataset(img_tensor, label_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Quantum Layer
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
        input_list = []
        w = self.weights.detach().cpu().numpy()
        for sample in x:
            input_angles = (sample[:self.n_qubits] * 2 * np.pi).detach().cpu().numpy()
            binding = dict(zip(self.data_params, input_angles)) | dict(zip(self.weight_params, w))
            circuits.append(self.qc.assign_parameters(binding))
            input_list.append(sample)

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

# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, num_classes=10, n_qubits=6):
        super().__init__()
        self.cnn = resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 64)
        self.dropout = nn.Dropout(0.5)
        self.qnn = QuantumLayer(n_qubits=n_qubits)
        self.fc = nn.Sequential(
            nn.Linear(64 + n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        cnn_out = self.dropout(self.cnn(x))
        q_out = self.qnn(cnn_out)
        combined = torch.cat((cnn_out, q_out), dim=1)
        return self.fc(combined)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

n_epochs = 30
train_losses, test_accs = [], []

# for epoch in range(n_epochs):
#     model.train()
#     total_loss = 0
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         out = model(imgs)
#         loss = criterion(out, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     train_losses.append(total_loss / len(train_loader))

#     # Evaluation
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for imgs, labels in test_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             out = model(imgs)
#             preds = torch.argmax(out, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     acc = 100 * correct / total
#     test_accs.append(acc)
#     print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Test Accuracy = {acc:.2f}%")

for epoch in range(start_epoch, n_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Evaluation
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
    test_accs.append(acc)

    print(f"📊 Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {acc:.2f}%")

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'test_accs': test_accs
    }, checkpoint_path)
    print(f"💾 Checkpoint saved at epoch {epoch+1}")

    scheduler.step()


# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label='Loss')
plt.title("Training Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accs, marker='o', label='Accuracy', color='green')
plt.title("Test Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("hybrid_qnn_results.png")
plt.show()
