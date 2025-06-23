import os
import json
import argparse
import numpy as np
import torch
import zipfile
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace, entropy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# === Enhanced Environment with Transfer Learning Support ===
class TransferMultiCircuitEnv(Env):
    """Enhanced quantum circuit environment supporting pre-trained weights"""
    def __init__(self, num_qubits: int = 3, max_steps: int = 15, target_state=None,
                 temperature: float = 1.0, reward_coeff: float = 1.0,
                 use_consciousness: bool = True, consciousness_weights: Dict[str, float] = None,
                 mode: str = 'custom', log_gate_history: bool = False, gate_history_logfile: Optional[str] = None):
        super().__init__()
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.temperature = temperature
        self.reward_coeff = reward_coeff
        self.use_consciousness = use_consciousness
        self.mode = mode
        self.log_gate_history = log_gate_history
        self.gate_history_logfile = gate_history_logfile

        # Default consciousness weights
        if consciousness_weights is None:
            consciousness_weights = {
                'memory_kernel': 0.5,
                'entanglement': 0.3,
                'exploration': 0.1,
                'transfer_bonus': 0.2
            }
        self.consciousness_weights = consciousness_weights

        self.step_count = 0
        self.gate_history = []
        self.last_fidelity = 0.0
        self.fidelity_history = []
        self.baseline_performance = 0.0

        self.actions = self._build_action_space()
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 ** num_qubits * 2,),
            dtype=np.float32
        )

        if target_state is None:
            target_state = [
                [0.40824829, 0.0], [-0.57735027, 0.0],
                [0.0, 0.0], [0.0, 0.0],
                [0.0, 0.0], [0.0, 0.0],
                [0.40824829, 0.0], [0.57735027, 0.0]
            ]
        if isinstance(target_state, (list, np.ndarray)):
            target_state = np.array(target_state, dtype=complex)
            target_state = target_state / np.linalg.norm(target_state)
        self.target_state = Statevector(target_state) if target_state is not None else self._get_default_target()
        self._reset_circuit()

    def _build_action_space(self) -> Dict[int, Tuple]:
        """Build action space matching your discovery code"""
        actions = {}
        action_id = 0
        max_qubits = 3  # Match your original code
        
        for i in range(max_qubits):
            actions[action_id] = ('h', i); action_id += 1
            actions[action_id] = ('x', i); action_id += 1
            # Match your RY gate parameterization
            for theta in np.linspace(np.pi/12, np.pi, 12):
                actions[action_id] = ('ry', (i, theta)); action_id += 1
        
        for i in range(max_qubits - 1):
            actions[action_id] = ('cx', (i, i + 1)); action_id += 1
        
        return actions

    def _get_default_target(self):
        """Get default target state based on mode"""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.num_qubits)
        
        if self.mode == 'ghz' or self.mode == 'custom':
            qc.h(0)
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        elif self.mode == 'bell':
            qc.h(0)
            qc.cx(0, 1)
        elif self.mode == 'w':
            if self.num_qubits == 3:
                qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0)
                qc.cx(0, 1)
                qc.ry(2 * np.arccos(1 / np.sqrt(2)), 1)
                qc.cx(1, 2)
            
        return Statevector.from_label('0' * self.num_qubits).evolve(qc)

    def _reset_circuit(self):
        """Reset quantum circuit to initial state"""
        from qiskit import QuantumCircuit
        self.qc = QuantumCircuit(self.num_qubits)
        self.step_count = 0
        self.last_fidelity = 0.0
        self.gate_history = []
        self.fidelity_history = []

    def get_observation(self) -> np.ndarray:
        """Get current quantum state as observation"""
        try:
            sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
            return np.concatenate([np.real(sv.data), np.imag(sv.data)]).astype(np.float32)
        except:
            return np.zeros(2 ** self.num_qubits * 2, dtype=np.float32)

    def memory_kernel_bonus(self, fidelity_history: List[float], gamma: float = 0.9) -> float:
        """Memory kernel consciousness module"""
        if not fidelity_history:
            return 0.0
        return sum(gamma**i * f for i, f in enumerate(reversed(fidelity_history)))

    def compute_entanglement_bonus(self, state: Statevector) -> float:
        """Entanglement awareness consciousness module"""
        if self.num_qubits <= 1:
            return 0.0
        try:
            reduced = partial_trace(state, [0])
            return entropy(reduced)
        except:
            return 0.0

    def transfer_learning_bonus(self) -> float:
        """Bonus for building incrementally (consciousness of prior knowledge)"""
        if len(self.fidelity_history) < 2:
            return 0.0
        
        # Reward consistent improvement (sign of building on prior knowledge)
        recent_trend = np.mean(np.diff(self.fidelity_history[-5:]))  # Recent improvement trend
        return max(0, recent_trend * 50)  # Positive bonus for improvement

    def step(self, action):
        """Execute one step in the environment"""
        action = int(action)
        if action not in self.actions:
            return self.get_observation(), -10.0, True, False, {'fidelity': 0.0, 'error': 'invalid_action'}

        gate, targets = self.actions[action]
        try:
            if gate == 'h':
                self.qc.h(targets)
            elif gate == 'x':
                self.qc.x(targets)
            elif gate == 'ry':
                self.qc.ry(targets[1], targets[0])
            elif gate == 'cx':
                self.qc.cx(*targets)
            self.gate_history.append(f'{gate}{targets}')
            reward_penalty = 0.0
            if len(self.gate_history) >= 2 and self.gate_history[-1] == self.gate_history[-2]:
                reward_penalty -= 0.5
        except Exception as e:
            return self.get_observation(), -10.0, True, False, {'fidelity': 0.0, 'error': str(e)}

        self.step_count += 1
        done = self.step_count >= self.max_steps

        try:
            sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
            fidelity = float(state_fidelity(sv, self.target_state))
            measured_fid = np.random.binomial(n=1000, p=min(1.0, fidelity)) / 1000.0
        except:
            fidelity = 0.0
            measured_fid = 0.0

        self.fidelity_history.append(measured_fid)
        base_reward = 200 * (measured_fid - self.last_fidelity) - 0.05 * len(self.gate_history)

        consciousness_bonus = 0.0
        entanglement_bonus = 0.0
        if self.use_consciousness:
            memory_bonus = self.memory_kernel_bonus(self.fidelity_history) * self.consciousness_weights['memory_kernel']
            entanglement_bonus = self.compute_entanglement_bonus(sv) * self.consciousness_weights['entanglement']
            transfer_bonus = self.transfer_learning_bonus() * self.consciousness_weights['transfer_bonus']
            exploration_bonus = np.random.normal(0, 0.1 / self.temperature) * self.consciousness_weights['exploration']
            consciousness_bonus = memory_bonus + entanglement_bonus + transfer_bonus + exploration_bonus
        else:
            entanglement_bonus = self.compute_entanglement_bonus(sv) * 10

        if self.use_consciousness:
            reward = (base_reward + consciousness_bonus) * self.reward_coeff
        else:
            reward = base_reward + entanglement_bonus
        if 'reward_penalty' in locals():
            reward += reward_penalty

        self.last_fidelity = measured_fid

        info = {
            'fidelity': measured_fid,
            'entanglement': entanglement_bonus,
            'gate_count': len(self.gate_history),
            'gate_count_penalty': -0.05 * len(self.gate_history),
            'consciousness_bonus': consciousness_bonus,
            'use_consciousness': self.use_consciousness
        }

        # Optionally log gate history at episode end
        if done and self.log_gate_history and self.gate_history_logfile:
            try:
                with open(self.gate_history_logfile, "a") as f:
                    f.write(json.dumps({"episode": getattr(self, "episode_counter", 0), "gates": self.gate_history}) + "\n")
            except Exception as e:
                pass
            self.episode_counter = getattr(self, "episode_counter", 0) + 1

        return self.get_observation(), reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._reset_circuit()
        return self.get_observation(), {}

# === Pre-trained Model Loading ===
def load_pretrained_model(model_path: str, env, device) -> Optional[PPO]:
    """Load pre-trained model from your discovery experiments"""
    try:
        if model_path.endswith('.zip'):
            # Load directly if it's a zip file
            model = PPO.load(model_path, env=env, device=device)
            print(f"✅ Loaded pre-trained model from {model_path}")
            return model
        else:
            print(f"Model file {model_path} not found")
            return None
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}")
        return None

def transfer_weights(source_model: PPO, target_model: PPO, transfer_rate: float = 1.0):
    """Transfer weights from source to target model"""
    try:
        source_state = source_model.policy.state_dict()
        target_state = target_model.policy.state_dict()
        
        # Transfer compatible layers
        transferred_layers = 0
        for name in target_state:
            if name in source_state and target_state[name].shape == source_state[name].shape:
                if transfer_rate == 1.0:
                    target_state[name] = source_state[name].clone()
                else:
                    # Partial transfer (blend with random initialization)
                    target_state[name] = (transfer_rate * source_state[name] + 
                                        (1 - transfer_rate) * target_state[name])
                transferred_layers += 1
        
        target_model.policy.load_state_dict(target_state)
        print(f"✅ Transferred {transferred_layers} layers with rate {transfer_rate}")
        return True
    except Exception as e:
        print(f"Weight transfer failed: {e}")
        return False

def evaluate_agent(model, env, episodes: int = 10, return_all: bool = False):
    """Evaluate agent, returning list of max fidelities per episode or full trajectories"""
    all_episode_fidelities = []
    for _ in range(episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        episode_fidelities = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False
            if isinstance(info, list):
                info = info[0]
            episode_fidelities.append(info.get('fidelity', 0.0))
            done = done or truncated
        if return_all:
            all_episode_fidelities.append(episode_fidelities)
        else:
            if episode_fidelities:
                all_episode_fidelities.append(max(episode_fidelities))
    return all_episode_fidelities


def run_consciousness_transfer_experiment(
    pretrained_model_paths: Dict[str, str],
    config: dict,
    use_consciousness: Optional[bool] = None,
    phases: int = 3,
    steps_per_phase: int = 20000,
    consciousness_weights: Optional[Dict[str, float]] = None,
    live_plot: bool = False,
    agent_names: Optional[List[str]] = None,
    log_gate_history: bool = False,
    models_export_dir: Optional[str] = None
):
    print("Quantum Circuit Consciousness with Transfer Learning")
    print("=" * 70)

    # Target state
    flat_target = np.array([complex(r, i) for r, i in config["target_state"]], dtype=complex)
    flat_target = flat_target / np.linalg.norm(flat_target)
    target_state = Statevector(flat_target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Agent configs
    agent_configs = []
    default_names = [
        "Consciousness + Transfer", "Baseline + Transfer",
        "Consciousness Fresh", "Baseline Fresh"
    ]
    if agent_names is not None and len(agent_names) == 4:
        default_names = agent_names
    for idx, (use_con, pretr) in enumerate([
        (True, "ghz"),
        (False, "ghz"),
        (True, None),
        (False, None)
    ]):
        if use_consciousness is not None:
            use_con = use_consciousness
        agent_configs.append({
            "use_consciousness": use_con,
            "name": default_names[idx],
            "pretrained": pretr
        })
    num_agents = len(agent_configs)
    models, envs, scores = [], [], [[] for _ in range(num_agents)]
    fidelity_trajectories = [[] for _ in range(num_agents)]

    print(f"\n🤖 Creating {num_agents} agents:")
    for i, agent_config in enumerate(agent_configs):
        env = DummyVecEnv([
            lambda agent_config=agent_config, temp_idx=i, global_cfg=config: TransferMultiCircuitEnv(
                num_qubits=global_cfg["num_qubits"],
                max_steps=global_cfg["max_steps"],
                target_state=flat_target,
                temperature=global_cfg["temperatures"][temp_idx % len(global_cfg["temperatures"])],
                use_consciousness=agent_config["use_consciousness"],
                consciousness_weights=consciousness_weights if consciousness_weights else None,
                mode='custom',
                log_gate_history=log_gate_history,
                gate_history_logfile=(
                    f"consciousness_transfer_results/gate_history_agent{i}.jsonl"
                    if log_gate_history else None
                )
            )
        ])
        model = PPO(
            "MlpPolicy", env, device=device,
            ent_coef=0.1, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128])
        )
        if agent_config["pretrained"] and agent_config["pretrained"] in pretrained_model_paths:
            pretrained_path = pretrained_model_paths[agent_config["pretrained"]]
            pretrained_model = load_pretrained_model(pretrained_path, env, device)
            if pretrained_model:
                transfer_weights(pretrained_model, model, transfer_rate=0.8)
                print(f"  Agent {i}: {agent_config['name']} (with {agent_config['pretrained']} weights)")
            else:
                print(f"  Agent {i}: {agent_config['name']} (failed to load pretrained)")
        else:
            print(f"  Agent {i}: {agent_config['name']} (fresh training)")
        models.append(model)
        envs.append(env)

    # Training phases
    all_phases = ["Fine-tuning Phase", "Advanced Learning Phase", "Mastery Phase"]
    if phases > len(all_phases):
        phase_names = [f"Phase {i+1}" for i in range(phases)]
    else:
        phase_names = all_phases[:phases]
    print(f"\n🚀 Training in {len(phase_names)} phases ({steps_per_phase} steps each)")

    # Live plotting setup
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        lines = [ax.plot([], [], label=agent_configs[i]["name"])[0] for i in range(num_agents)]
        ax.set_xlabel("Phase")
        ax.set_ylabel("Median Fidelity")
        ax.set_title("Live Training Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show(block=False)

    # Training loop
    for phase_idx, phase_name in enumerate(phase_names):
        print(f"\n📈 {phase_name} ({phase_idx + 1}/{len(phase_names)})")
        # Adjust learning parameters per phase
        if phase_idx == 0:
            learning_rate = 1e-4
        elif phase_idx == 1:
            learning_rate = 5e-5
        else:
            learning_rate = 1e-5
        for model in models:
            model.learning_rate = learning_rate
        # Train each agent
        for i, (model, agent_config) in enumerate(zip(models, agent_configs)):
            print(f"  Training Agent {i} ({agent_config['name']})...")
            model.learn(total_timesteps=steps_per_phase, reset_num_timesteps=False)
            fidelities = evaluate_agent(model, envs[i], episodes=config["evaluation_episodes"])
            scores[i].append(np.median(fidelities))
            # Save full trajectory for CSV
            full_fids = evaluate_agent(model, envs[i], episodes=config["evaluation_episodes"], return_all=True)
            fidelity_trajectories[i].append(full_fids)
            print(f"    Median Fidelity: {scores[i][-1]:.4f}")
        # Live plot update
        if live_plot:
            for idx, line in enumerate(lines):
                line.set_data(range(len(scores[idx])), scores[idx])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.1)

    if live_plot:
        plt.ioff()

    os.makedirs("consciousness_transfer_results", exist_ok=True)

    # Save fidelity trajectories per agent to CSV
    for i, agent_config in enumerate(agent_configs):
        csv_path = f"consciousness_transfer_results/agent_{i}_fidelity_trajectory.csv"
        with open(csv_path, "w") as f:
            f.write("phase,episode,max_fidelity,full_episode_fidelities\n")
            for phase_idx, phase_fids in enumerate(fidelity_trajectories[i]):
                for ep_idx, ep_fids in enumerate(phase_fids):
                    max_fid = max(ep_fids) if ep_fids else 0.0
                    f.write(f"{phase_idx},{ep_idx},{max_fid},\"{ep_fids}\"\n")

    # Export models
    if models_export_dir is None:
        models_export_dir = "consciousness_transfer_results/models"
    os.makedirs(models_export_dir, exist_ok=True)
    for i, model in enumerate(models):
        model_dir = os.path.join(models_export_dir, f"agent_{i}")
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, "model.zip"))

    # Results analysis (unchanged)
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green', 'orange']
    for i, (scores_list, agent_config) in enumerate(zip(scores, agent_configs)):
        plt.plot(scores_list, 'o-', color=colors[i], linewidth=2, markersize=8,
                label=agent_config['name'])
    plt.xlabel("Training Phase")
    plt.ylabel("Fidelity")
    plt.title("Training Evolution Across Phases")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(phase_names)), [f"Phase {i+1}" for i in range(len(phase_names))])
    plt.subplot(2, 3, 2)
    final_scores = [scores[i][-1] if scores[i] else 0.0 for i in range(num_agents)]
    agent_names = [config['name'] for config in agent_configs]
    bars = plt.bar(range(num_agents), final_scores, color=colors[:num_agents], alpha=0.7)
    plt.ylabel("Final Fidelity")
    plt.title("Final Performance Comparison")
    plt.xticks(range(num_agents), [f"Agent {i}" for i in range(num_agents)], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    for bar, score, name in zip(bars, final_scores, agent_names):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.subplot(2, 3, 3)
    transfer_scores = [scores[0][-1], scores[1][-1]]
    fresh_scores = [scores[2][-1], scores[3][-1]]
    x_pos = np.arange(2)
    width = 0.35
    plt.bar(x_pos - width/2, transfer_scores, width, label='With Transfer',
           color=['red', 'blue'], alpha=0.7)
    plt.bar(x_pos + width/2, fresh_scores, width, label='Fresh Training',
           color=['lightcoral', 'lightblue'], alpha=0.7)
    plt.ylabel("Final Fidelity")
    plt.title("Transfer Learning Effectiveness")
    plt.xticks(x_pos, ['Consciousness', 'Baseline'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.subplot(2, 3, 4)
    consciousness_scores = [scores[0][-1], scores[2][-1]]
    baseline_scores = [scores[1][-1], scores[3][-1]]
    x_pos = np.arange(2)
    plt.bar(x_pos - width/2, consciousness_scores, width, label='Consciousness',
           color='red', alpha=0.7)
    plt.bar(x_pos + width/2, baseline_scores, width, label='Baseline',
           color='blue', alpha=0.7)
    plt.ylabel("Final Fidelity")
    plt.title("Consciousness vs Baseline")
    plt.xticks(x_pos, ['With Transfer', 'Fresh Training'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.subplot(2, 3, 5)
    transfer_improvement = [(scores[0][-1] - scores[2][-1]), (scores[1][-1] - scores[3][-1])]
    consciousness_improvement = [(scores[0][-1] - scores[1][-1]), (scores[2][-1] - scores[3][-1])]
    x_pos = np.arange(2)
    plt.bar(x_pos - width/2, transfer_improvement, width, label='Transfer Learning Boost',
           color='green', alpha=0.7)
    plt.bar(x_pos + width/2, consciousness_improvement, width, label='Consciousness Boost',
           color='purple', alpha=0.7)
    plt.ylabel("Fidelity Improvement")
    plt.title("Improvement Analysis")
    plt.xticks(x_pos, ['Consciousness', 'Baseline'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.subplot(2, 3, 6)
    for i, (scores_list, agent_config) in enumerate(zip(scores, agent_configs)):
        if len(scores_list) > 1:
            improvements = np.diff(scores_list)
            stability = -np.std(improvements)
            plt.bar(i, stability, color=colors[i], alpha=0.7)
    plt.ylabel("Learning Stability (higher is better)")
    plt.title("Learning Curve Stability")
    plt.xticks(range(num_agents), [f"Agent {i}" for i in range(num_agents)], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("consciousness_transfer_results/complete_analysis.png",
               dpi=300, bbox_inches='tight')
    plt.show()

    # Save results
    results = {
        "agent_configs": agent_configs,
        "final_scores": final_scores,
        "training_history": [[float(f) for f in scores[i]] for i in range(num_agents)],
        "transfer_improvement": transfer_improvement,
        "consciousness_improvement": consciousness_improvement
    }
    with open("consciousness_transfer_results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING + CONSCIOUSNESS EXPERIMENT COMPLETE!")
    print("=" * 70)
    print(f"\n FINAL RESULTS:")
    for i, (score, config) in enumerate(zip(final_scores, agent_configs)):
        print(f"  Agent {i} ({config['name']}): {score:.4f}")
    best_agent = np.argmax(final_scores)
    print(f"\n Best Agent: {agent_configs[best_agent]['name']} (Agent {best_agent})")
    transfer_helps = (scores[0][-1] + scores[1][-1]) / 2 > (scores[2][-1] + scores[3][-1]) / 2
    consciousness_helps = (scores[0][-1] + scores[2][-1]) / 2 > (scores[1][-1] + scores[3][-1]) / 2
    print(f"\n Transfer Learning Helps: {'YES' if transfer_helps else ' NO'}")
    print(f"Consciousness Helps: {'YES' if consciousness_helps else ' NO'}")
    if consciousness_helps and transfer_helps:
        print(f"\n BREAKTHROUGH: Both consciousness AND transfer learning improve performance!")
    elif consciousness_helps:
        print(f"\n Consciousness provides advantage even without transfer learning!")
    elif transfer_helps:
        print(f"\n Transfer learning works, but consciousness needs refinement!")
    else:
        print(f"\n Interesting negative result - need to investigate further!")

    # Results analysis
    os.makedirs("consciousness_transfer_results", exist_ok=True)
    
    # Create comprehensive visualization
    plt.figure(figsize=(20, 12))
    
    # 1. Training evolution
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green', 'orange']
    for i, (scores_list, agent_config) in enumerate(zip(scores, agent_configs)):
        plt.plot(scores_list, 'o-', color=colors[i], linewidth=2, markersize=8,
                label=agent_config['name'])
    plt.xlabel("Training Phase")
    plt.ylabel("Fidelity")
    plt.title("Training Evolution Across Phases")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(phase_names)), [f"Phase {i+1}" for i in range(len(phase_names))])
    
    # 2. Final performance comparison
    plt.subplot(2, 3, 2)
    final_scores = [scores[i][-1] if scores[i] else 0.0 for i in range(num_agents)]
    agent_names = [config['name'] for config in agent_configs]
    
    bars = plt.bar(range(num_agents), final_scores, color=colors[:num_agents], alpha=0.7)
    plt.ylabel("Final Fidelity")
    plt.title("Final Performance Comparison")
    plt.xticks(range(num_agents), [f"Agent {i}" for i in range(num_agents)], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score, name in zip(bars, final_scores, agent_names):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Transfer learning effectiveness
    plt.subplot(2, 3, 3)
    transfer_scores = [scores[0][-1], scores[1][-1]]  # Agents with transfer
    fresh_scores = [scores[2][-1], scores[3][-1]]    # Agents without transfer
    
    x_pos = np.arange(2)
    width = 0.35
    
    plt.bar(x_pos - width/2, transfer_scores, width, label='With Transfer', 
           color=['red', 'blue'], alpha=0.7)
    plt.bar(x_pos + width/2, fresh_scores, width, label='Fresh Training', 
           color=['lightcoral', 'lightblue'], alpha=0.7)
    
    plt.ylabel("Final Fidelity")
    plt.title("Transfer Learning Effectiveness")
    plt.xticks(x_pos, ['Consciousness', 'Baseline'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Consciousness vs Baseline (with and without transfer)
    plt.subplot(2, 3, 4)
    consciousness_scores = [scores[0][-1], scores[2][-1]]  # Consciousness agents
    baseline_scores = [scores[1][-1], scores[3][-1]]       # Baseline agents
    
    x_pos = np.arange(2)
    plt.bar(x_pos - width/2, consciousness_scores, width, label='Consciousness', 
           color='red', alpha=0.7)
    plt.bar(x_pos + width/2, baseline_scores, width, label='Baseline', 
           color='blue', alpha=0.7)
    
    plt.ylabel("Final Fidelity")
    plt.title("Consciousness vs Baseline")
    plt.xticks(x_pos, ['With Transfer', 'Fresh Training'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 5. Improvement analysis
    plt.subplot(2, 3, 5)
    transfer_improvement = [(scores[0][-1] - scores[2][-1]), (scores[1][-1] - scores[3][-1])]
    consciousness_improvement = [(scores[0][-1] - scores[1][-1]), (scores[2][-1] - scores[3][-1])]
    
    x_pos = np.arange(2)
    plt.bar(x_pos - width/2, transfer_improvement, width, label='Transfer Learning Boost', 
           color='green', alpha=0.7)
    plt.bar(x_pos + width/2, consciousness_improvement, width, label='Consciousness Boost', 
           color='purple', alpha=0.7)
    
    plt.ylabel("Fidelity Improvement")
    plt.title("Improvement Analysis")
    plt.xticks(x_pos, ['Consciousness', 'Baseline'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Learning curve smoothness
    plt.subplot(2, 3, 6)
    for i, (scores_list, agent_config) in enumerate(zip(scores, agent_configs)):
        if len(scores_list) > 1:
            improvements = np.diff(scores_list)
            stability = -np.std(improvements)  # Negative std (higher is more stable)
            plt.bar(i, stability, color=colors[i], alpha=0.7)
    
    plt.ylabel("Learning Stability (higher is better)")
    plt.title("Learning Curve Stability")
    plt.xticks(range(num_agents), [f"Agent {i}" for i in range(num_agents)], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("consciousness_transfer_results/complete_analysis.png", 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        "agent_configs": agent_configs,
        "final_scores": final_scores,
        "training_history": [[float(f) for f in scores[i]] for i in range(num_agents)],
        "transfer_improvement": transfer_improvement,
        "consciousness_improvement": consciousness_improvement
    }
    
    with open("consciousness_transfer_results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING + CONSCIOUSNESS EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    print(f"\n FINAL RESULTS:")
    for i, (score, config) in enumerate(zip(final_scores, agent_configs)):
        print(f"  Agent {i} ({config['name']}): {score:.4f}")
    
    # Analysis
    best_agent = np.argmax(final_scores)
    print(f"\n Best Agent: {agent_configs[best_agent]['name']} (Agent {best_agent})")
    
    transfer_helps = (scores[0][-1] + scores[1][-1]) / 2 > (scores[2][-1] + scores[3][-1]) / 2
    consciousness_helps = (scores[0][-1] + scores[2][-1]) / 2 > (scores[1][-1] + scores[3][-1]) / 2
    
    print(f"\n Transfer Learning Helps: {'YES' if transfer_helps else ' NO'}")
    print(f"Consciousness Helps: {'YES' if consciousness_helps else ' NO'}")
    
    if consciousness_helps and transfer_helps:
        print(f"\n BREAKTHROUGH: Both consciousness AND transfer learning improve performance!")
    elif consciousness_helps:
        print(f"\n Consciousness provides advantage even without transfer learning!")
    elif transfer_helps:
        print(f"\n Transfer learning works, but consciousness needs refinement!")
    else:
        print(f"\n Interesting negative result - need to investigate further!")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)

def parse_pretrained_arg(pretrained_list):
    """Parse --pretrained bell=path.zip style arguments"""
    result = {}
    for entry in pretrained_list:
        if "=" in entry:
            k, v = entry.split("=", 1)
            result[k.strip()] = v.strip()
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Circuit Consciousness Transfer Experiment")
    parser.add_argument("--pretrained", nargs="*", default=[],
                        help="Pretrained model paths, e.g. --pretrained bell=path/to/bell.zip ghz=path/to/ghz.zip")
    parser.add_argument("--consciousness", dest="consciousness", action="store_true", help="Enable consciousness modules")
    parser.add_argument("--no-consciousness", dest="consciousness", action="store_false", help="Disable consciousness modules")
    parser.set_defaults(consciousness=None)
    parser.add_argument("--config", default="config.json", help="Config file (JSON)")
    parser.add_argument("--phases", type=int, default=3, help="Number of training phases")
    parser.add_argument("--steps-per-phase", type=int, default=20000, help="Steps per phase")
    parser.add_argument("--consciousness-weights", type=str, default=None, help="JSON string or file for consciousness weights")
    parser.add_argument("--live-plot", action="store_true", help="Enable live plotting of training curves")
    parser.add_argument("--log-gate-history", action="store_true", help="Log full gate history per episode")
    parser.add_argument("--models-export-dir", type=str, default=None, help="Directory to export trained models")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Fallback default config
        config = {
            "num_qubits": 3,
            "max_steps": 15,
            "temperatures": [0.5, 1.0, 2.0, 4.0],
            "target_state": [
                [0.40824829, 0.0], [-0.57735027, 0.0],
                [0.0, 0.0], [0.0, 0.0],
                [0.0, 0.0], [0.0, 0.0],
                [0.40824829, 0.0], [0.57735027, 0.0]
            ],
            "timesteps_per_phase": args.steps_per_phase,
            "evaluation_episodes": 10
        }

    # Pretrained models
    pretrained_paths = parse_pretrained_arg(args.pretrained)
    if not pretrained_paths:
        # Check default files
        default_paths = {
            "bell": "ppo_bell_agent_finetuned.zip",
            "ghz": "ppo_ghz_agent_finetuned.zip",
            "w": "ppo_w_agent_finetuned.zip"
        }
        for name, path in default_paths.items():
            if os.path.exists(path):
                pretrained_paths[name] = path
    # Validate files exist
    valid_paths = {}
    for name, path in pretrained_paths.items():
        if os.path.exists(path):
            valid_paths[name] = path
            print(f"Found {name} model: {path}")
        else:
            print(f"Missing {name} model: {path}")
    if not valid_paths:
        print("No pre-trained models found. Please provide with --pretrained bell=path.zip etc.")
        exit(1)

    # Consciousness weights
    consciousness_weights = None
    if args.consciousness_weights:
        if os.path.exists(args.consciousness_weights):
            with open(args.consciousness_weights, "r") as f:
                consciousness_weights = json.load(f)
        else:
            try:
                consciousness_weights = json.loads(args.consciousness_weights)
            except Exception:
                print("Could not parse --consciousness-weights argument")
                consciousness_weights = None

    run_consciousness_transfer_experiment(
        valid_paths,
        config,
        use_consciousness=args.consciousness,
        phases=args.phases,
        steps_per_phase=args.steps_per_phase,
        consciousness_weights=consciousness_weights,
        live_plot=args.live_plot,
        log_gate_history=args.log_gate_history,
        models_export_dir=args.models_export_dir
    )
