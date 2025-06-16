import os
import json
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
                 mode: str = 'custom'):
        super().__init__()
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.temperature = temperature
        self.reward_coeff = reward_coeff
        self.use_consciousness = use_consciousness
        self.mode = mode
        
        # Default consciousness weights
        if consciousness_weights is None:
            consciousness_weights = {
                'memory_kernel': 0.5,
                'entanglement': 0.3,
                'exploration': 0.1,
                'transfer_bonus': 0.2  # New: bonus for building on pre-learned knowledge
            }
        self.consciousness_weights = consciousness_weights
        
        self.step_count = 0
        self.gate_history = []
        self.last_fidelity = 0.0
        self.fidelity_history = []
        self.baseline_performance = 0.0  # Track performance without consciousness

        # Build comprehensive action space (matching your discovery code)
        self.actions = self._build_action_space()
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(2 ** num_qubits * 2,), 
            dtype=np.float32
        )

        # Set target state
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
        
        # Apply quantum gate (matching your discovery code)
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
            
        except Exception as e:
            return self.get_observation(), -10.0, True, False, {'fidelity': 0.0, 'error': str(e)}
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Calculate fidelity
        try:
            sv = Statevector.from_label('0' * self.num_qubits).evolve(self.qc)
            fidelity = float(state_fidelity(sv, self.target_state))
        except:
            fidelity = 0.0
        
        self.fidelity_history.append(fidelity)
        
        # Base reward (matching your discovery code)
        base_reward = 200 * (fidelity - self.last_fidelity)
        
        # Consciousness bonuses (only if enabled)
        consciousness_bonus = 0.0
        entanglement_bonus = 0.0
        
        if self.use_consciousness:
            # Memory kernel bonus
            memory_bonus = self.memory_kernel_bonus(self.fidelity_history) * self.consciousness_weights['memory_kernel']
            
            # Entanglement bonus  
            entanglement_bonus = self.compute_entanglement_bonus(sv) * self.consciousness_weights['entanglement']
            
            # Transfer learning bonus (consciousness of prior knowledge)
            transfer_bonus = self.transfer_learning_bonus() * self.consciousness_weights['transfer_bonus']
            
            # Temperature-based exploration bonus
            exploration_bonus = np.random.normal(0, 0.1 / self.temperature) * self.consciousness_weights['exploration']
            
            consciousness_bonus = memory_bonus + entanglement_bonus + transfer_bonus + exploration_bonus
        else:
            # For baseline: add basic entanglement bonus (matching your discovery code)
            entanglement_bonus = self.compute_entanglement_bonus(sv) * 10  # Match your original scaling
        
        # Final reward
        if self.use_consciousness:
            reward = (base_reward + consciousness_bonus) * self.reward_coeff
        else:
            reward = base_reward + entanglement_bonus  # Baseline matching your discovery code
        
        self.last_fidelity = fidelity
        
        info = {
            'fidelity': fidelity,
            'entanglement': entanglement_bonus,
            'gate_count': len(self.gate_history),
            'consciousness_bonus': consciousness_bonus,
            'use_consciousness': self.use_consciousness
        }
        
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

# === Enhanced Experiment with Transfer Learning ===
def run_consciousness_transfer_experiment(pretrained_model_paths: Dict[str, str]):
    """Run consciousness experiment using pre-trained models as starting points"""
    
    print("Quantum Circuit Consciousness with Transfer Learning")
    print("=" * 70)
    
    # Configuration
    config = {
        "num_qubits": 3,
        "max_steps": 15,
        "temperatures": [0.5, 1.0, 2.0, 4.0],
        "target_state": [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],  # |000⟩ + |111⟩ (GHZ-like)
        "timesteps_per_phase": 20000,
        "evaluation_episodes": 10
    }
    
    # Convert target state
    flat_target = np.array([complex(r, i) for r, i in config["target_state"]], dtype=complex)
    flat_target = flat_target / np.linalg.norm(flat_target)
    target_state = Statevector(flat_target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Agent configurations
    agent_configs = [
        {"use_consciousness": True, "name": "Consciousness + Transfer", "pretrained": "ghz"},
        {"use_consciousness": False, "name": "Baseline + Transfer", "pretrained": "ghz"},
        {"use_consciousness": True, "name": "Consciousness Fresh", "pretrained": None},
        {"use_consciousness": False, "name": "Baseline Fresh", "pretrained": None}
    ]

    num_agents = len(agent_configs)
    models, envs, scores = [], [], [[] for _ in range(num_agents)]
    
    print(f"\n🤖 Creating {num_agents} agents:")

    # Create agents
    for i, agent_config in enumerate(agent_configs):
        # Create environment
        env = DummyVecEnv([lambda agent_config=agent_config, temp_idx=i, global_cfg=config: TransferMultiCircuitEnv(
            num_qubits=global_cfg["num_qubits"],
            max_steps=global_cfg["max_steps"],
            target_state=flat_target,
            temperature=global_cfg["temperatures"][temp_idx % len(global_cfg["temperatures"])],
            use_consciousness=agent_config["use_consciousness"],
            mode='custom'
        )])
        
        # Create model
        model = PPO("MlpPolicy", env, device=device, 
                   ent_coef=0.1, verbose=0,
                   policy_kwargs=dict(net_arch=[128, 128]))  # Match your discovery architecture
        
        # Load pre-trained weights if specified
        if agent_config["pretrained"] and agent_config["pretrained"] in pretrained_model_paths:
            pretrained_path = pretrained_model_paths[agent_config["pretrained"]]
            pretrained_model = load_pretrained_model(pretrained_path, env, device)
            
            if pretrained_model:
                transfer_weights(pretrained_model, model, transfer_rate=0.8)  # 80% transfer
                print(f"  Agent {i}: {agent_config['name']} (with {agent_config['pretrained']} weights)")
            else:
                print(f"  Agent {i}: {agent_config['name']} (failed to load pretrained)")
        else:
            print(f"  Agent {i}: {agent_config['name']} (fresh training)")
        
        models.append(model)
        envs.append(env)

    # Training phases
    phases = ["Fine-tuning Phase", "Advanced Learning Phase", "Mastery Phase"]
    timesteps_per_phase = config["timesteps_per_phase"]
    
    print(f"\n🚀 Training in {len(phases)} phases ({timesteps_per_phase} steps each)")

    # Training loop
    for phase_idx, phase_name in enumerate(phases):
        print(f"\n📈 {phase_name} ({phase_idx + 1}/{len(phases)})")
        
        # Adjust learning parameters per phase
        if phase_idx == 0:  # Fine-tuning
            learning_rate = 1e-4
        elif phase_idx == 1:  # Advanced learning
            learning_rate = 5e-5
        else:  # Mastery
            learning_rate = 1e-5
        
        # Update learning rates
        for model in models:
            model.learning_rate = learning_rate
        
        # Train each agent
        for i, (model, agent_config) in enumerate(zip(models, agent_configs)):
            print(f"  Training Agent {i} ({agent_config['name']})...")
            
            # Train
            model.learn(total_timesteps=timesteps_per_phase, reset_num_timesteps=False)
            
            # Evaluate
            fidelities = []
            for episode in range(config["evaluation_episodes"]):
                obs, _ = envs[i].reset()  # Unpack reset return value
                done = False
                episode_fidelities = []
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = envs[i].step(action)
                    episode_fidelities.append(info.get('fidelity', 0.0))
                    done = done or truncated

                if episode_fidelities:
                    fidelities.append(max(episode_fidelities))
            
            mean_fidelity = np.mean(fidelities)
            std_fidelity = np.std(fidelities)
            scores[i].append(mean_fidelity)
            
            print(f"    Performance: {mean_fidelity:.4f} ± {std_fidelity:.4f}")

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
    plt.xticks(range(len(phases)), [f"Phase {i+1}" for i in range(len(phases))])
    
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

if __name__ == "__main__":
    # Example usage - specify paths to your pre-trained models
    pretrained_paths = {
        "bell": "ppo_bell_agent_finetuned.zip",      # Replace with actual paths
        "ghz": "ppo_ghz_agent_finetuned.zip",        # to your trained models
        "w": "ppo_w_agent_finetuned.zip"
    }
    
    print("Please update the pretrained_paths dictionary with your actual model file paths")
    print("Looking for .zip files from your quantum circuit discovery experiments")
    
    # Check if paths exist
    valid_paths = {}
    for name, path in pretrained_paths.items():
        if os.path.exists(path):
            valid_paths[name] = path
            print(f"Found {name} model: {path}")
        else:
            print(f"Missing {name} model: {path}")
    
    if valid_paths:
        run_consciousness_transfer_experiment(valid_paths)
    else:
        print(" No pre-trained models found. Please:")
        print("   1. Update the pretrained_paths dictionary with correct file paths")
        print("   2. Ensure your .zip model files are accessible")
        print("   3. Run the experiment again")
