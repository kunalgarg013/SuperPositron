import os
import json
import numpy as np
import torch
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace, entropy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
from copy import deepcopy
import matplotlib.pyplot as plt

# === SoftQFT utilities ===
def compute_identity_coherence_loss(old_weights, new_weights):
    drift = 0.0
    count = 0
    for name in old_weights:
        if name in new_weights:
            drift += torch.norm(old_weights[name] - new_weights[name]).item()
            count += 1
    return drift / max(count, 1)

def memory_kernel_bonus(fidelity_history, gamma=0.9):
    return sum(gamma**i * f for i, f in enumerate(reversed(fidelity_history)))

def compute_entanglement_bonus(state: Statevector, trace_qubits: list = [0]) -> float:
    try:
        reduced = partial_trace(state, trace_qubits)
        return entropy(reduced)
    except:
        return 0.0

# === Environment ===
class MultiCircuitEnv(Env):
    def __init__(self, num_qubits=3, max_steps=15, target_state=None):
        super().__init__()
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.step_count = 0
        self.gate_history = []
        self.last_fidelity = 0.0
        self.fidelity_history = []

        self.actions = {}
        aid = 0
        for i in range(num_qubits):
            self.actions[aid] = ('h', i); aid += 1
            self.actions[aid] = ('x', i); aid += 1
            self.actions[aid] = ('z', i); aid += 1
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                self.actions[aid] = ('ry', (i, theta)); aid += 1
                self.actions[aid] = ('rz', (i, theta)); aid += 1
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                self.actions[aid] = ('cx', (i, j)); aid += 1

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 ** num_qubits * 2,), dtype=np.float32)

        self.target_state = Statevector(target_state) if isinstance(target_state, (list, np.ndarray)) else target_state
        self._reset_circuit()

    def _reset_circuit(self):
        from qiskit import QuantumCircuit
        self.qc = QuantumCircuit(self.num_qubits)
        self.step_count = 0
        self.last_fidelity = 0.0
        self.gate_history = []
        self.fidelity_history = []

    def get_observation(self):
        sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
        return np.concatenate([np.real(sv.data), np.imag(sv.data)]).astype(np.float32)

    def step(self, action):
        action = int(action)
        if action not in self.actions:
            return self.get_observation(), -5.0, True, False, {'fidelity': -1.0}
        gate, target = self.actions[action]
        try:
            if gate == 'h': self.qc.h(target)
            elif gate == 'x': self.qc.x(target)
            elif gate == 'z': self.qc.z(target)
            elif gate == 'ry': self.qc.ry(target[1], target[0])
            elif gate == 'rz': self.qc.rz(target[1], target[0])
            elif gate == 'cx': self.qc.cx(*target)
            self.gate_history.append(f'{gate}{target}')
        except:
            return self.get_observation(), -5.0, True, False, {'fidelity': -1.0}
        self.step_count += 1
        done = self.step_count >= self.max_steps
        sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
        fidelity = state_fidelity(sv, self.target_state)
        self.fidelity_history.append(fidelity)
        bonus = memory_kernel_bonus(self.fidelity_history)
        ent_bonus = compute_entanglement_bonus(sv)
        reward = (fidelity - self.last_fidelity) * 200 + 0.5 * bonus + 0.3 * ent_bonus
        self.last_fidelity = fidelity
        return self.get_observation(), reward, done, False, {'fidelity': fidelity, 'entanglement': ent_bonus}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_circuit()
        return self.get_observation(), {}

# === Training logic ===
def train_with_identity_regularization(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    flat_target = np.array([complex(r, i) for r, i in cfg['target_state']])
    flat_target /= np.linalg.norm(flat_target)

    env = DummyVecEnv([lambda: MultiCircuitEnv(cfg['num_qubits'], cfg['max_steps'], flat_target)])

    model = PPO("MlpPolicy", env, ent_coef=0.1, verbose=1)
    if "pretrained_path" in cfg and os.path.exists(cfg["pretrained_path"]):
        pretrained = PPO.load(cfg["pretrained_path"], env=env)
        model.policy.load_state_dict(pretrained.policy.state_dict())
        print(f"Loaded pretrained weights from {cfg['pretrained_path']}")

    drift_log = []
    fidelities = []

    for _ in range(cfg['rounds']):
        old_weights = {n: p.clone() for n, p in model.policy.named_parameters()}
        model.learn(total_timesteps=cfg['timesteps'])
        new_weights = {n: p for n, p in model.policy.named_parameters()}

        drift = compute_identity_coherence_loss(old_weights, new_weights)
        drift_log.append(drift)

        obs = env.reset()
        _, _, done, _, info = env.step(env.envs[0].action_space.sample())
        fidelities.append(info['fidelity'])

    # Save results
    os.makedirs("pt_identity", exist_ok=True)
    plt.plot(fidelities, label="Fidelity")
    plt.plot(drift_log, label="Coherence Drift")
    plt.xlabel("Training Rounds")
    plt.title("Identity Coherence vs Fidelity")
    plt.legend()
    plt.savefig("pt_identity/coherence_vs_fidelity.png")
    plt.close()

if __name__ == "__main__":
    train_with_identity_regularization("identity_config.json")
