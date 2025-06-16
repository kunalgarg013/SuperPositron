import os
import json
import numpy as np
import torch
from qiskit.quantum_info import Statevector, state_fidelity
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
from copy import deepcopy
import matplotlib.pyplot as plt

class MultiCircuitEnv(Env):
    def __init__(self, num_qubits=3, max_steps=15, target_state=None):
        super().__init__()
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.step_count = 0
        self.gate_history = []
        self.last_fidelity = 0.0
        self.actions = {}
        action_id = 0
        for i in range(num_qubits):
            self.actions[action_id] = ('h', i); action_id += 1
            self.actions[action_id] = ('x', i); action_id += 1
            for theta in np.linspace(np.pi/12, np.pi, 12):
                self.actions[action_id] = ('ry', (i, theta)); action_id += 1
        for i in range(num_qubits - 1):
            self.actions[action_id] = ('cx', (i, i + 1)); action_id += 1

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2**num_qubits * 2,), dtype=np.float32)
        self.target_state = Statevector(target_state) if isinstance(target_state, (list, np.ndarray)) else target_state
        self._reset_circuit()

    def _reset_circuit(self):
        from qiskit import QuantumCircuit
        self.qc = QuantumCircuit(self.num_qubits)
        self.step_count = 0
        self.last_fidelity = 0.0
        self.gate_history = []

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
        except:
            return self.get_observation(), -5.0, True, False, {'fidelity': -1.0}
        self.step_count += 1
        done = self.step_count >= self.max_steps
        sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
        fidelity = state_fidelity(sv, self.target_state)
        reward = (fidelity - self.last_fidelity) * 200
        self.last_fidelity = fidelity
        return self.get_observation(), reward, done, False, {'fidelity': fidelity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_circuit()
        return self.get_observation(), {}

def swap_models(model_a, model_b):
    for p1, p2 in zip(model_a.policy.parameters(), model_b.policy.parameters()):
        p1.data, p2.data = deepcopy(p2.data), deepcopy(p1.data)

def smooth_fidelity(fids, alpha=0.2):
    smoothed = [fids[0]]
    for f in fids[1:]:
        smoothed.append(alpha * f + (1 - alpha) * smoothed[-1])
    return smoothed

def main():
    with open("parallel_tempering_config.json", "r") as f:
        cfg = json.load(f)

    num_agents = len(cfg["temperatures"])
    envs, models, fidelities = [], [], [[] for _ in range(num_agents)]
    coherence_drift = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flat_target = [complex(r, i) for r, i in cfg["target_state"]]
    flat_target = np.array(flat_target)
    flat_target /= np.linalg.norm(flat_target)

    for i in range(num_agents):
        env = DummyVecEnv([lambda: MultiCircuitEnv(cfg["num_qubits"], cfg["max_steps"], flat_target)])
        model = PPO("MlpPolicy", env, device=device, ent_coef=0.1, verbose=0)
        models.append(model)
        envs.append(env)

    swap_every = cfg["swap_interval"]
    steps_per_agent = cfg["timesteps_per_agent"]

    for step in range(0, steps_per_agent, swap_every):
        for i in range(num_agents):
            models[i].learn(total_timesteps=swap_every, reset_num_timesteps=False)
        for i in range(num_agents):
            obs = envs[i].reset()
            _, reward, _, _, info = envs[i].envs[0].step(envs[i].envs[0].action_space.sample())
            fidelities[i].append(info["fidelity"])

            if i == 0:
                # Identity coherence tracking (simplified: norm diff on shared layer)
                try:
                    with torch.no_grad():
                        shared = models[i].policy.mlp_extractor.shared_net[0].weight
                        drift = torch.norm(shared - shared.clone()).item()
                        coherence_drift.append(drift)
                except Exception:
                    coherence_drift.append(0.0)

        for i in range(num_agents - 1):
            if np.random.rand() < 0.5:
                swap_models(models[i], models[i+1])
                print(f"Swapped models {i} and {i+1}")

    os.makedirs("pt_logs", exist_ok=True)
    for i, fids in enumerate(fidelities):
        plt.plot(smooth_fidelity(fids), label=f"T={cfg['temperatures'][i]}")
    if coherence_drift:
        plt.plot(smooth_fidelity(coherence_drift), '--k', label="Agent 0 Drift")
    plt.xlabel("Swap Rounds")
    plt.ylabel("Fidelity / Drift")
    plt.title("Fidelity Evolution + Identity Coherence Drift")
    plt.legend()
    plt.grid(True)
    plt.savefig("pt_logs/pt_identity_coherence_experiment.png")
    plt.close()

if __name__ == "__main__":
    main()