import numpy as np
import os
import torch
import argparse
import json
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector, partial_trace, entropy
from stable_baselines3 import PPO
from gymnasium import Env, spaces

FINE_TUNE = True
FINE_TUNE_TIMESTEPS = 100_000
FINE_TUNE_LR = 1e-4
FINE_TUNE_ENT_COEF = 0.01
USE_TENSORBOARD = False
TENSORBOARD_LOG_DIR = "./ppo_logs/"


def initialize_cuda():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        return torch.device("cpu")
    try:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        return torch.device("cpu")


class MultiCircuitEnv(Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, mode='ghz', num_qubits=3, max_steps=12, custom_target_state=None):
        super().__init__()
        assert mode in ['ghz', 'bell', 'w', 'custom'], "Unsupported mode!"
        self.mode = mode
        self.num_qubits = 2 if mode == 'bell' else num_qubits
        self.max_steps = max_steps
        self.step_count = 0
        self.last_fidelity = 0
        self.gate_history = []
        self.fidelity_evolution = []
        self.custom_target_state = custom_target_state

        self.actions = {}
        action_id = 0
        max_qubits = 3
        for i in range(max_qubits):
            self.actions[action_id] = ('h', i); action_id += 1
            self.actions[action_id] = ('x', i); action_id += 1
            for theta in np.linspace(np.pi/12, np.pi, 12):
                self.actions[action_id] = ('ry', (i, theta)); action_id += 1
        for i in range(max_qubits - 1):
            self.actions[action_id] = ('cx', (i, i + 1)); action_id += 1

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 ** self.num_qubits * 2,), dtype=np.float32)

        self.target_state = self._get_target_state()
        self._reset_circuit()

    def _get_target_state(self):
        if self.mode == 'custom' and self.custom_target_state is not None:
            return self.custom_target_state
        qc = QuantumCircuit(self.num_qubits)
        if self.mode == 'ghz':
            qc.h(0)
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        elif self.mode == 'bell':
            qc.h(0); qc.cx(0, 1)
        elif self.mode == 'w':
            if self.num_qubits == 3:
                qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0)
                qc.cx(0, 1)
                qc.ry(2 * np.arccos(1 / np.sqrt(2)), 1)
                qc.cx(1, 2)
            else:
                raise NotImplementedError("W state only implemented for 3 qubits.")
        return Statevector.from_label('0' * self.num_qubits).evolve(qc)

    def _reset_circuit(self):
        self.qc = QuantumCircuit(self.num_qubits)
        self.step_count = 0
        self.last_fidelity = 0
        self.gate_history = []
        self.fidelity_evolution = []

    def get_observation(self):
        sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
        return np.concatenate([np.real(sv.data), np.imag(sv.data)]).astype(np.float32)

    def step(self, action):
        action = int(action)
        if action not in self.actions:
            return self.get_observation(), -5.0, True, False, {'fidelity': -1.0}

        gate, targets = self.actions[action]
        try:
            if gate == 'h': self.qc.h(targets)
            elif gate == 'x': self.qc.x(targets)
            elif gate == 'ry': self.qc.ry(targets[1], targets[0])
            elif gate == 'cx': self.qc.cx(*targets)
            self.gate_history.append(f'{gate}{targets}')
        except Exception:
            return self.get_observation(), -5.0, True, False, {'fidelity': -1.0}

        self.step_count += 1
        done = self.step_count >= self.max_steps

        sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
        fidelity = state_fidelity(sv, self.target_state)
        self.fidelity_evolution.append(fidelity)
        entanglement_bonus = 0.0
        if self.num_qubits > 1:
            try:
                reduced = partial_trace(sv, [0])
                entanglement_bonus = entropy(reduced)
            except Exception:
                entanglement_bonus = 0.0

        reward = 200 * (fidelity - self.last_fidelity) + 10 * entanglement_bonus
        self.last_fidelity = fidelity

        return self.get_observation(), reward, done, False, {'fidelity': fidelity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_circuit()
        return self.get_observation(), {}

    def render(self):
        print(self.qc.draw())

    def plot_fidelity_curve(self):
        plt.figure()
        plt.plot(self.fidelity_evolution, marker='o')
        plt.title('Fidelity Evolution During Execution')
        plt.xlabel('Step')
        plt.ylabel('Fidelity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("fidelity_plot.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Circuit RL Discovery Tool")
    parser.add_argument(
    "--mode", 
    type=str, 
    choices=["curriculum", "custom"], 
    default="curriculum",
    help="Choose training mode: 'curriculum' (bell→ghz→w) or 'custom'"
    )
    parser.add_argument("--config", type=str, help="Path to custom target config file")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training timesteps")
    args = parser.parse_args()


    device = initialize_cuda()

    with open(args.config, 'r') as f:
        config = json.load(f)
        vec = [complex(p[0], p[1]) for p in config['target_state']]
        vec = np.array(vec)
        vec /= np.linalg.norm(vec)
        target = Statevector(vec)

    env = MultiCircuitEnv(mode='custom', num_qubits=config['num_qubits'], max_steps=config['max_steps'], custom_target_state=target)
    model = PPO("MlpPolicy", env, verbose=1, device=device, ent_coef=0.1, policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=args.timesteps)

    obs, _ = env.reset()
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step+1}: {env.gate_history[-1]} | Fidelity: {info['fidelity']:.4f}")
        if done:
            break

    print("Final Fidelity:", info['fidelity'])
    print("Gate Sequence:", env.gate_history)
    env.plot_fidelity_curve()
