import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from mlxtend.data import loadlocal_mnist
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer.primitives import Sampler

# ----- Configuration -----
n_qubits = 6
checkpoint_path = "checkpoint_epoch_latest.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load Test Data -----
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

# ----- Quantum Layer -----
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

# ----- Hybrid Model -----
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

# ----- Load Model -----
model = HybridModel().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----- Run Inference -----
all_preds, all_probs, all_targets, all_imgs = [], [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(confs.cpu().tolist())
        all_targets.extend(labels.tolist())
        all_imgs.extend(imgs.cpu())

# ----- Analyze Predictions -----
results = list(zip(all_imgs, all_preds, all_probs, all_targets))
correct = [r for r in results if r[1] == r[3]]
incorrect = [r for r in results if r[1] != r[3]]

top_correct = sorted(correct, key=lambda x: -x[2])[:5]
top_incorrect = sorted(incorrect, key=lambda x: -x[2])[:5]
lowconf_correct = sorted(correct, key=lambda x: x[2])[:5]

# ----- Plot Function -----
def save_images(title, examples, filename):
    fig, axs = plt.subplots(1, len(examples), figsize=(15, 3))
    fig.suptitle(title, fontsize=16)
    for i, (img, pred, conf, true) in enumerate(examples):
        axs[i].imshow(img.squeeze().numpy(), cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"P:{pred}, T:{true}\n{conf*100:.1f}%")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"📸 Saved: {filename}")

# ----- Display Results -----
save_images("✅ Most Confident Correct Predictions", top_correct, "top_correct_preds.png")
save_images("❌ Most Confident Incorrect Predictions", top_incorrect, "top_incorrect_preds.png")
save_images("🤔 Least Confident Correct Predictions", lowconf_correct, "lowconf_correct_preds.png")

