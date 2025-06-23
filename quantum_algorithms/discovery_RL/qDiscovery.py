import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector, partial_trace, entropy
from stable_baselines3 import PPO
from gymnasium import Env, spaces

# Fine-tuning parameters for improving fidelity
FINE_TUNE = True
FINE_TUNE_TIMESTEPS = 50000
FINE_TUNE_LR = 1e-5
FINE_TUNE_ENT_COEF = 0.001

class MultiCircuitEnv(Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, mode='ghz', num_qubits=3, max_steps=15):
        super().__init__()
        assert mode in ['ghz', 'bell', 'w'], "Unsupported mode!"
        self.mode = mode
        self.num_qubits = 2 if mode == 'bell' else num_qubits
        self.max_steps = max_steps
        self.step_count = 0
        self.last_fidelity = 0
        self.gate_history = []

        # Define fixed action space size for all modes
        self.actions = {}
        action_id = 0
        
        # Basic gates for all qubits
        max_qubits = 3  # Maximum number of qubits across all modes
        for i in range(max_qubits):
            self.actions[action_id] = ('h', i); action_id += 1
            self.actions[action_id] = ('x', i); action_id += 1
            # Fixed set of rotation angles
            for theta in [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]:
                self.actions[action_id] = ('ry', (i, theta))
                action_id += 1
    
        # CNOT gates between adjacent qubits
        for i in range(max_qubits - 1):
            self.actions[action_id] = ('cx', (i, i + 1)); action_id += 1

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 ** self.num_qubits * 2,), dtype=np.float32)

        self.target_state = self._get_target_state()
        self._reset_circuit()

    def _get_target_state(self):
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
        # if fidelity > 0.99:
        #     reward = 100.0
        #     done = True

        return self.get_observation(), reward, done, False, {'fidelity': fidelity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_circuit()
        return self.get_observation(), {}

    def render(self):
        print(self.qc.draw())


if __name__ == "__main__":
    curriculum = ['bell', 'ghz', 'w']
    for mode in curriculum:
        print(f"=== Starting Training for Mode: {mode.upper()} ===")
        model_path = f"ppo_{mode}_agent"
        env = MultiCircuitEnv(mode=mode, num_qubits=3, max_steps=10)

        if os.path.exists(model_path + ".zip"):
            print(" Loading existing model...")
            model = PPO.load(model_path, env=env)
        else:
            print(" Training new model...")
            model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.1)
            model.learn(total_timesteps=200_000)
            model.save(model_path)

        # Fine-tuning phase to push fidelity higher if enabled
        if FINE_TUNE:
            print("Fine-tuning the model for higher fidelity...")
            model = PPO.load(model_path, env=env, learning_rate=FINE_TUNE_LR, ent_coef=FINE_TUNE_ENT_COEF)
            model.learn(total_timesteps=FINE_TUNE_TIMESTEPS)
            finetuned_path = model_path + "_finetuned"
            model.save(finetuned_path)
            print(f"Fine-tuning completed and saved to {finetuned_path}")
        obs, _ = env.reset()
        print("\n=== Executing Learned Circuit ===")
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {step+1} | Action: {env.gate_history[-1]}, Fidelity: {info['fidelity']:.4f}, Reward: {reward:.3f}")
            env.render()
            if done:
                print("Early stop: Target fidelity reached!")
                break

        final_state = Statevector.from_label('0' * env.num_qubits).evolve(env.qc)
        print("\n=== Final Diagnostic ===")
        print("Final Statevector:\n", final_state)
        print("Target Statevector:\n", env.target_state)
        print("State Fidelity:", state_fidelity(final_state, env.target_state))
        if env.num_qubits > 1:
            reduced = partial_trace(final_state, [0])
            print("Entanglement Entropy (tracing out q0):", entropy(reduced))
        print("Gate Sequence:", env.gate_history)