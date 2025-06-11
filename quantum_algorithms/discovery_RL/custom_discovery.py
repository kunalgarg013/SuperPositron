import numpy as np
import os
import torch

# Add CUDA initialization code after imports
def initialize_cuda():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        return torch.device("cpu")

    try:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return device
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        return torch.device("cpu")

from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector, partial_trace, entropy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Env, spaces

# Fine-tuning parameters for improving fidelity
FINE_TUNE = True
FINE_TUNE_TIMESTEPS = 100_000
FINE_TUNE_LR = 1e-4
FINE_TUNE_ENT_COEF = 0.01

class MultiCircuitEnv(Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, mode='ghz', num_qubits=3, max_steps=12, custom_target_state=None):
        super().__init__()
        assert mode in ['ghz', 'bell', 'w', 'custom'], "Unsupported mode!"
        self.mode = mode
        self.num_qubits = num_qubits if mode != 'bell' else 2
        self.max_steps = max_steps
        self.step_count = 0
        self.last_fidelity = 0
        self.gate_history = []
        self.custom_target_state = custom_target_state

        # Define fixed action space
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
        entanglement_bonus = 0.0

        if self.num_qubits > 1:
            try:
                reduced = partial_trace(sv, [0])
                entanglement_bonus = entropy(reduced)
            except Exception:
                entanglement_bonus = 0.0

        reward = (fidelity - self.last_fidelity) * 200 + entanglement_bonus
        self.last_fidelity = fidelity

        return self.get_observation(), reward, done, False, {'fidelity': fidelity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_circuit()
        return self.get_observation(), {}

    def render(self):
        print(self.qc.draw())

if __name__ == "__main__":
    device = initialize_cuda()

    # === Test Block for Custom State Search ===
    print("\n=== Custom State Training ===")
    custom_qc = QuantumCircuit(3)
    custom_qc.h(0)
    custom_qc.cx(0, 1)
    custom_qc.ry(np.pi/3, 2)
    custom_qc.cx(1, 2)
    target_state = Statevector.from_label('000').evolve(custom_qc)

    custom_env = MultiCircuitEnv(mode='custom', num_qubits=3, max_steps=10, custom_target_state=target_state)
    custom_model = PPO("MlpPolicy", custom_env, verbose=1, device=device, ent_coef=0.1,
                      policy_kwargs=dict(net_arch=[128, 128]),
                      tensorboard_log="./ppo_logs_custom/")
    custom_model.learn(total_timesteps=500_000)

    obs, _ = custom_env.reset()
    print("\n=== Executing Learned Circuit for Custom State ===")
    for step in range(custom_env.max_steps):
        action, _ = custom_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = custom_env.step(action)
        print(f"Step {step+1} | Action: {custom_env.gate_history[-1]}, Fidelity: {info['fidelity']:.4f}, Reward: {reward:.3f}")
        custom_env.render()
        if done:
            print("Early stop: Target fidelity reached!")
            break

    final_state = Statevector.from_label('0' * custom_env.num_qubits).evolve(custom_env.qc)
    print("\n=== Final Diagnostic for Custom State ===")
    print("Final Statevector:\n", final_state)
    print("Target Statevector:\n", custom_env.target_state)
    print("State Fidelity:", state_fidelity(final_state, custom_env.target_state))
    if custom_env.num_qubits > 1:
        reduced = partial_trace(final_state, [0])
        print("Entanglement Entropy (tracing out q0):", entropy(reduced))
    print("Gate Sequence:", custom_env.gate_history)
