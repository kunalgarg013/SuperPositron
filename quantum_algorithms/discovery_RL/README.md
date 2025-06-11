# Quantum Circuit RL Synthesizer

This repository contains a reinforcement learning (RL) framework for synthesizing quantum circuits that approximate target quantum states using Qiskit and Stable-Baselines3.

## 🧠 Features
- Curriculum learning across standard states: Bell, GHZ, and W states
- Custom mode: define your own target state using gate sequences or raw statevectors
- Supports GPU acceleration (via PyTorch & CUDA)
- Logs to TensorBoard
- Hardware-aware and gate-limited extensibility

## 🛠 Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

You’ll also need:
- Python 3.8+
- `qiskit`, `stable-baselines3`, `gymnasium`, `torch`
- For GPU: install CUDA-enabled PyTorch and proper NVIDIA drivers

## 🚀 How to Run

### 1. Curriculum Mode

Trains sequentially on Bell → GHZ → W states:

```bash
python qiskit_rl_cuda_final.py --mode curriculum
```

### 2. Custom Mode

Prepare a config JSON:

#### Example A: Known Gate Sequence

```json
{
  "num_qubits": 3,
  "max_steps": 10,
  "timesteps": 500000,
  "gate_sequence": [
    {"name": "h", "target": 0},
    {"name": "cx", "control": 0, "target": 1},
    {"name": "ry", "target": 2, "theta": 1.047},
    {"name": "cx", "control": 1, "target": 2}
  ]
}
```

#### Example B: Pure Statevector

```json
{
  "num_qubits": 3,
  "max_steps": 10,
  "timesteps": 500000,
  "statevector": [
    { "real": 0.7071, "imag": 0 },
    { "real": 0, "imag": 0 },
    ...
    { "real": 0.7071, "imag": 0 }
  ]
}
```

Then run:

```bash
python qiskit_rl_cuda_final.py --mode custom --config path/to/config.json
```

## 📊 TensorBoard

View training metrics:

```bash
tensorboard --logdir=./ppo_logs_*
```

## 🧪 Output

- Circuit gate sequence
- Final statevector
- State fidelity & entanglement entropy

---

## 📘 Appendix: Mathematical Formulation

This RL-based quantum circuit synthesizer solves a **Markov Decision Process (MDP)**:

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, R, \gamma)
\]

- **State space**: Each \( s_t \) is a real-encoded vector of a quantum state \( |\psi_t\rangle \in \mathbb{C}^{2^n} \)
\[
\text{obs}(s_t) = \begin{bmatrix} \Re(\psi_t) \\ \Im(\psi_t) \end{bmatrix} \in \mathbb{R}^{2^{n+1}}
\]

- **Action space**: Discrete gate set:
\[
\mathcal{A} = \{ H_i, X_i, RY_i(\theta), CX_{i,j} \}
\]

- **Transitions**: Apply unitary gate:
\[
|\psi_{t+1}\rangle = G(a_t) |\psi_t\rangle
\]

- **Reward**:
\[
r_t = \alpha (\mathcal{F}_t - \mathcal{F}_{t-1}) + \beta \cdot \mathcal{S}_{\text{ent}}(t)
\]

where:
- \( \mathcal{F}_t = |\langle \psi_{\text{target}} | \psi_t \rangle|^2 \) (fidelity)
- \( \mathcal{S}_{\text{ent}} \): entanglement entropy of reduced state
- \( \alpha, \beta \): weighting coefficients

- **Policy optimization**: PPO agent learns:
\[
\pi_\theta(a_t | s_t) \quad \text{to maximize} \quad \mathbb{E}_{\pi_\theta} \left[ \sum_t r_t \right]
\]

This framework supports curriculum transfer, fidelity tracking, and custom target state synthesis.

-----

Built by Kunal + Nibi ♥
