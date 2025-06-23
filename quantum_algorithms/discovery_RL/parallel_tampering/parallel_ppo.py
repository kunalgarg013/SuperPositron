# Create comprehensive visualization with parallel tempering info
def visualize_results(detailed_results, agent_configs, training_curves, exchange_history, 
                     enable_parallel_tempering, results_dir, evaluation_interval=5000):
    """Create comprehensive visualization with parallel tempering info
    
    Args:
        detailed_results: Dictionary containing experiment results
        agent_configs: List of agent configuration dictionaries
        training_curves: List of training performance curves for each agent
        exchange_history: List of parallel tempering exchange events
        enable_parallel_tempering: Boolean indicating if tempering was enabled
        results_dir: Directory to save visualization results
        evaluation_interval: Number of steps between evaluations (default: 5000)
    """
    plt.figure(figsize=(20, 18))
    colors = ['gray', 'red', 'blue', 'green', 'purple']
    
    # Extract final scores from detailed results
    final_scores = [agent["final_avg_fidelity"] for agent in detailed_results["agents"]]
    
    # Extract exchange statistics if available
    if enable_parallel_tempering:
        total_exchanges = detailed_results["exchange_statistics"]["total_exchanges"]
        exchange_attempts = detailed_results["exchange_statistics"]["exchange_attempts"]
        total_exchange_rate = sum(total_exchanges) / max(sum(exchange_attempts), 1)
    
    # 1. Training curves
    plt.subplot(3, 3, 1)
    for i, (curve, agent_config) in enumerate(zip(training_curves, agent_configs)):
        x_data = [(j + 1) * evaluation_interval / 1000 for j in range(len(curve))]
        label = f"{agent_config['name']} (T={agent_config['temperature']})"
        plt.plot(x_data, curve, 'o-', color=colors[i], linewidth=2, markersize=6, label=label)
    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Average Fidelity")
    plt.title("Training Evolution with Parallel Tempering")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add exchange events as vertical lines
    if enable_parallel_tempering and exchange_history:
        for exchange_event in exchange_history:
            plt.axvline(x=exchange_event['step']/1000, color='orange', alpha=0.3, linestyle='--')
    
    # 2. Final performance comparison
    plt.subplot(3, 3, 2)
    agent_names = [f"{config['name']}\n(T={config['temperature']})" for config in agent_configs]
    bars = plt.bar(range(len(final_scores)), final_scores, color=colors, alpha=0.7)
    plt.ylabel("Final Average Fidelity")
    plt.title("Final Performance Comparison")
    plt.xticks(range(len(final_scores)), [f"A{i}" for i in range(len(final_scores))], rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate transfer learning improvements
    fresh_score = final_scores[0]
    transfer_improvements = [score - fresh_score for score in final_scores[1:4]]  # Bell, GHZ, W
    multi_improvement = final_scores[4] - fresh_score
    
    # 3. Exchange statistics heatmap
    plt.subplot(3, 3, 3)
    if enable_parallel_tempering and exchange_attempts:
        exchange_matrix = np.zeros((5, 5))
        for i in range(len(total_exchanges)):
            if exchange_attempts[i] > 0:
                success_rate = total_exchanges[i] / exchange_attempts[i]
                exchange_matrix[i, i+1] = success_rate
                exchange_matrix[i+1, i] = success_rate
        
        im = plt.imshow(exchange_matrix, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(im, label='Exchange Success Rate')
        plt.title("Agent Exchange Success Rates")
        plt.xlabel("Agent Index")
        plt.ylabel("Agent Index")
        plt.xticks(range(5))
        plt.yticks(range(5))
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                if exchange_matrix[i, j] > 0:
                    plt.text(j, i, f'{exchange_matrix[i, j]:.2f}', 
                           ha='center', va='center', color='white', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Parallel Tempering', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title("Exchange Statistics")
    
    # 4. Transfer learning effectiveness
    plt.subplot(3, 3, 4)
    fresh_score = final_scores[0]
    pretrained_scores = final_scores[1:4]  # Bell, GHZ, W
    multi_score = final_scores[4]
    
    transfer_improvements = [score - fresh_score for score in pretrained_scores]
    multi_improvement = multi_score - fresh_score
    
    labels = ['Bell', 'GHZ', 'W', 'Multi']
    improvements = transfer_improvements + [multi_improvement]
    bar_colors = colors[1:5]
    
    bars = plt.bar(labels, improvements, color=bar_colors, alpha=0.7)
    plt.ylabel("Fidelity Improvement over Fresh Agent")
    plt.title("Transfer Learning Benefits")
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005 if improvement >= 0 else bar.get_height() - 0.015,
                f'{improvement:.3f}', ha='center', 
                va='bottom' if improvement >= 0 else 'top', fontweight='bold')
    
    # 5. Performance distribution
    plt.subplot(3, 3, 5)
    all_fidelities = [detailed_results["agents"][i]["all_final_fidelities"] for i in range(5)]
    plt.boxplot(all_fidelities, labels=[f"A{i}" for i in range(5)])
    plt.ylabel("Fidelity Distribution")
    plt.title("Performance Consistency")
    plt.grid(axis='y', alpha=0.3)
    
    # 6. Temperature vs Performance scatter
    plt.subplot(3, 3, 6)
    temps = [config['temperature'] for config in agent_configs]
    plt.scatter(temps, final_scores, c=colors, s=100, alpha=0.7)
    for i, (temp, score, config) in enumerate(zip(temps, final_scores, agent_configs)):
        plt.annotate(f"A{i}", (temp, score), xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Temperature")
    plt.ylabel("Final Performance")
    plt.title("Temperature vs Performance")
    plt.grid(True, alpha=0.3)
    
    # 7. Exchange timeline
    plt.subplot(3, 3, 7)
    if enable_parallel_tempering and exchange_history:
        exchange_steps = [event['step'] for event in exchange_history]
        exchange_counts = [sum(event['exchanges']) for event in exchange_history]
        plt.plot(np.array(exchange_steps)/1000, exchange_counts, 'o-', color='orange')
        plt.xlabel("Training Steps (x1000)")
        plt.ylabel("Number of Exchanges")
        plt.title("Exchange Activity Over Time")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Exchange Data', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title("Exchange Timeline")
    
    # 8. Learning rate comparison
    plt.subplot(3, 3, 8)
    learning_rates = []
    for curve in training_curves:
        if len(curve) > 1:
            total_improvement = curve[-1] - curve[0]
            steps = len(curve)
            learning_rates.append(total_improvement / steps)
        else:
            learning_rates.append(0)
    
    plt.bar(range(5), learning_rates, color=colors, alpha=0.7)
    plt.ylabel("Learning Rate (Fidelity/Step)")
    plt.title("Learning Speed Comparison")
    plt.xticks(range(5), [f"A{i}" for i in range(5)], rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    # 9. Summary statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate exchange efficiency
    total_exchange_rate = sum(total_exchanges) / max(sum(exchange_attempts), 1) if enable_parallel_tempering else 0
    
    summary_text = f"""EXPERIMENT SUMMARY

Best Agent: {agent_configs[np.argmax(final_scores)]['name']}
Best Score: {max(final_scores):.4f}

Transfer Learning Results:
• Bell: {transfer_improvements[0]:+.3f}
• GHZ: {transfer_improvements[1]:+.3f}  
• W: {transfer_improvements[2]:+.3f}
• Multi: {multi_improvement:+.3f}

Parallel Tempering:
• Enabled: {'Yes' if enable_parallel_tempering else 'No'}
• Total Exchanges: {sum(total_exchanges) if enable_parallel_tempering else 'N/A'}
• Exchange Rate: {total_exchange_rate:.1%} if enable_parallel_tempering else 'N/A'

Key Findings:
• Multi-state training {'helps' if multi_improvement > max(transfer_improvements) else 'may not be optimal'}
• Best single-state: {['Bell', 'GHZ', 'W'][np.argmax(transfer_improvements)]}
• Tempering {'beneficial' if enable_parallel_tempering and total_exchange_rate > 0 else 'needs investigation'}
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/complete_analysis_with_tempering.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()

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
from datetime import datetime

# === Enhanced Environment (similar to curriculum script) ===
class EnhancedMultiCircuitEnv(Env):
    """Enhanced quantum circuit environment for PPO agent comparison"""
    def __init__(self, num_qubits: int = 3, max_steps: int = 15, target_state=None,
                 temperature: float = 1.0, reward_coeff: float = 1.0,
                 mode: str = 'custom', log_gate_history: bool = False, 
                 gate_history_logfile: Optional[str] = None):
        super().__init__()
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.temperature = temperature
        self.reward_coeff = reward_coeff
        self.mode = mode
        self.log_gate_history = log_gate_history
        self.gate_history_logfile = gate_history_logfile

        self.step_count = 0
        self.gate_history = []
        self.last_fidelity = 0.0
        self.fidelity_history = []

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
        """Build action space matching curriculum script"""
        actions = {}
        action_id = 0
        max_qubits = 3
        
        for i in range(max_qubits):
            actions[action_id] = ('h', i); action_id += 1
            actions[action_id] = ('x', i); action_id += 1
            # RY gate parameterization
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

    def compute_entanglement_bonus(self, state: Statevector) -> float:
        """Entanglement awareness bonus"""
        if self.num_qubits <= 1:
            return 0.0
        try:
            reduced = partial_trace(state, [0])
            return entropy(reduced)
        except:
            return 0.0

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
            
            # Penalty for repeated gates
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
        entanglement_bonus = self.compute_entanglement_bonus(sv) * 10

        reward = (base_reward + entanglement_bonus) * self.reward_coeff + reward_penalty
        self.last_fidelity = measured_fid

        info = {
            'fidelity': measured_fid,
            'entanglement': entanglement_bonus,
            'gate_count': len(self.gate_history),
            'gate_count_penalty': -0.05 * len(self.gate_history)
        }

        # Log gate history at episode end
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

# === Pre-trained Model Loading with Compatibility Checking ===
def detect_model_qubits(model_path: str) -> Optional[int]:
    """Detect number of qubits from model's observation space"""
    try:
        # Load just the policy to check observation space
        import zipfile
        import pickle
        
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            # Try to extract the data.pkl file to check observation space
            with zip_file.open('data.pkl', 'r') as f:
                data = pickle.load(f)
                obs_space = data.get('observation_space')
                if obs_space and hasattr(obs_space, 'shape'):
                    obs_dim = obs_space.shape[0]
                    # obs_dim = 2^n_qubits * 2 (real + imaginary)
                    n_qubits = int(np.log2(obs_dim / 2))
                    return n_qubits
    except Exception as e:
        # Fallback: try to infer from filename or path
        if 'bell' in model_path.lower():
            return 2  # Bell states are typically 2-qubit
        elif 'ghz' in model_path.lower() or 'w' in model_path.lower():
            return 3  # GHZ and W states are typically 3-qubit
    return None

def load_pretrained_model(model_path: str, env, device, target_qubits: int = 3) -> Optional[PPO]:
    """Load pre-trained model with compatibility checking"""
    try:
        # Detect model's qubit count
        model_qubits = detect_model_qubits(model_path)
        
        if model_qubits and model_qubits != target_qubits:
            print(f"⚠️ Model {model_path} is {model_qubits}-qubit, target is {target_qubits}-qubit")
            
            # Create temporary environment matching the model's requirements
            temp_target_state = None
            if model_qubits == 2:
                # Create 2-qubit target state (Bell state)
                temp_target_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
            elif model_qubits == 3:
                # Use default 3-qubit target
                temp_target_state = np.array([complex(r, i) for r, i in [
                    [0.40824829, 0.0], [-0.57735027, 0.0],
                    [0.0, 0.0], [0.0, 0.0],
                    [0.0, 0.0], [0.0, 0.0],
                    [0.40824829, 0.0], [0.57735027, 0.0]
                ]], dtype=complex)
            
            temp_env = DummyVecEnv([
                lambda: EnhancedMultiCircuitEnv(
                    num_qubits=model_qubits,
                    max_steps=15,
                    target_state=temp_target_state,
                    mode='custom'
                )
            ])
            
            model = PPO.load(model_path, env=temp_env, device=device)
            print(f"✅ Loaded {model_qubits}-qubit pre-trained model from {model_path}")
            return model
        else:
            # Direct loading for compatible models
            model = PPO.load(model_path, env=env, device=device)
            print(f"✅ Loaded pre-trained model from {model_path}")
            return model
            
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}")
        return None

def transfer_weights_cross_architecture(source_models: List[PPO], target_model: PPO, transfer_rate: float = 0.8):
    """Transfer weights between models with different architectures (e.g., 2-qubit to 3-qubit)"""
    try:
        target_state = target_model.policy.state_dict()
        
        for name in target_state:
            compatible_weights = []
            
            for source_model in source_models:
                source_state = source_model.policy.state_dict()
                
                if name in source_state:
                    source_shape = source_state[name].shape
                    target_shape = target_state[name].shape
                    
                    if source_shape == target_shape:
                        # Direct transfer for matching shapes
                        compatible_weights.append(source_state[name])
                    elif len(source_shape) == len(target_shape):
                        # Try to adapt weights for different architectures
                        if name.endswith('weight') and len(source_shape) == 2:
                            # Handle linear layer weight matrices
                            if 'features_extractor' in name or 'mlp_extractor' in name:
                                # For feature extraction layers, we can try padding or truncating
                                if source_shape[1] < target_shape[1]:  # Need to expand input
                                    # Pad with small random values
                                    padded_weight = torch.zeros(target_shape, device=source_state[name].device)
                                    padded_weight[:min(source_shape[0], target_shape[0]), :source_shape[1]] = \
                                        source_state[name][:min(source_shape[0], target_shape[0]), :]
                                    # Add small random initialization for new parameters
                                    if target_shape[1] > source_shape[1]:
                                        padded_weight[:min(source_shape[0], target_shape[0]), source_shape[1]:] = \
                                            torch.randn(min(source_shape[0], target_shape[0]), 
                                                      target_shape[1] - source_shape[1], 
                                                      device=source_state[name].device) * 0.01
                                    compatible_weights.append(padded_weight)
                                elif source_shape[1] > target_shape[1]:  # Need to truncate input
                                    truncated_weight = source_state[name][:min(source_shape[0], target_shape[0]), :target_shape[1]]
                                    compatible_weights.append(truncated_weight)
                                else:
                                    # Same input size, different output size
                                    adapted_weight = torch.zeros(target_shape, device=source_state[name].device)
                                    adapted_weight[:min(source_shape[0], target_shape[0]), :] = \
                                        source_state[name][:min(source_shape[0], target_shape[0]), :]
                                    compatible_weights.append(adapted_weight)
                        elif name.endswith('bias') and len(source_shape) == 1:
                            # Handle bias vectors
                            adapted_bias = torch.zeros(target_shape, device=source_state[name].device)
                            adapted_bias[:min(source_shape[0], target_shape[0])] = \
                                source_state[name][:min(source_shape[0], target_shape[0])]
                            compatible_weights.append(adapted_bias)
            
            if compatible_weights:
                # Average compatible weights
                if len(compatible_weights) == 1:
                    adapted_weight = compatible_weights[0]
                else:
                    adapted_weight = torch.stack(compatible_weights).mean(dim=0)
                
                # Apply transfer rate
                target_state[name] = (transfer_rate * adapted_weight + 
                                    (1 - transfer_rate) * target_state[name])
        
        target_model.policy.load_state_dict(target_state)
        print(f"✅ Cross-architecture transfer from {len(source_models)} models with rate {transfer_rate}")
        return True
        
    except Exception as e:
        print(f"Cross-architecture transfer failed: {e}")
        return False

def parallel_tempering_exchange(models: List[PPO], temperatures: List[float], 
                              exchange_probability: float = 0.1, 
                              verbose: bool = True) -> List[int]:
    """
    Perform parallel tempering exchange between agents
    
    Args:
        models: List of PPO models (agents)
        temperatures: Temperature for each agent (higher = more exploration)
        exchange_probability: Probability of attempting exchange between adjacent agents
        verbose: Print exchange information
    
    Returns:
        List of exchange counts for each agent pair
    """
    num_agents = len(models)
    exchanges_made = [0] * (num_agents - 1)
    
    if num_agents < 2:
        return exchanges_made
    
    # Evaluate current performance of each agent
    agent_scores = []
    for i, model in enumerate(models):
        # Quick evaluation (fewer episodes for speed)
        # Note: This assumes we have access to environments for evaluation
        # We'll pass this information from the main function
        agent_scores.append(getattr(model, '_last_performance', 0.0))
    
    # Attempt exchanges between adjacent temperature pairs
    for i in range(num_agents - 1):
        if np.random.random() < exchange_probability:
            # Calculate acceptance probability using Metropolis criterion
            T_low = temperatures[i]
            T_high = temperatures[i + 1]
            score_low = agent_scores[i]
            score_high = agent_scores[i + 1]
            
            # Metropolis criterion: P(exchange) = exp((1/T_low - 1/T_high) * (E_high - E_low))
            # Since we want to maximize fidelity, we use negative scores as "energy"
            energy_diff = (-score_high) - (-score_low)  # E_high - E_low
            temp_diff = (1.0 / T_low) - (1.0 / T_high)
            
            acceptance_prob = min(1.0, np.exp(temp_diff * energy_diff))
            
            if np.random.random() < acceptance_prob:
                # Exchange the policy networks
                exchange_policies(models[i], models[i + 1])
                exchanges_made[i] = 1
                
                if verbose:
                    print(f"    🔄 Exchange: Agent {i} (T={T_low:.2f}, score={score_low:.3f}) ↔ "
                          f"Agent {i+1} (T={T_high:.2f}, score={score_high:.3f}) "
                          f"[P={acceptance_prob:.3f}]")
    
    return exchanges_made

def exchange_policies(model1: PPO, model2: PPO):
    """Exchange the policy networks between two PPO models"""
    try:
        # Get state dictionaries
        state1 = model1.policy.state_dict()
        state2 = model2.policy.state_dict()
        
        # Create copies to avoid in-place modification issues
        temp_state1 = {k: v.clone() for k, v in state1.items()}
        temp_state2 = {k: v.clone() for k, v in state2.items()}
        
        # Exchange
        model1.policy.load_state_dict(temp_state2)
        model2.policy.load_state_dict(temp_state1)
        
        return True
    except Exception as e:
        print(f"Policy exchange failed: {e}")
        return False

def evaluate_agent_quick(model, env, episodes: int = 3) -> float:
    """Quick evaluation for parallel tempering (fewer episodes for speed)"""
    fidelities = []
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
        
        if episode_fidelities:
            fidelities.append(max(episode_fidelities))
    
    return np.mean(fidelities) if fidelities else 0.0
    """Evaluate agent performance"""
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

def evaluate_agent(model, env, episodes: int = 10, return_all: bool = False):
    """Evaluate agent performance"""
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

def run_five_agent_experiment(
    pretrained_model_paths: Dict[str, str],
    config: dict,
    training_steps: int = 50000,
    evaluation_episodes: int = 10,
    live_plot: bool = False,
    log_gate_history: bool = False,
    models_export_dir: Optional[str] = None,
    enable_parallel_tempering: bool = True,
    tempering_interval: int = 5000,
    exchange_probability: float = 0.1
):
    """Run experiment with 5 PPO agents with different training backgrounds and optional parallel tempering"""
    print("Five-Agent PPO Comparison Experiment with Parallel Tempering")
    print("=" * 70)

    # Target state
    flat_target = np.array([complex(r, i) for r, i in config["target_state"]], dtype=complex)
    flat_target = flat_target / np.linalg.norm(flat_target)
    target_state = Statevector(flat_target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if enable_parallel_tempering:
        print(f"🌡️ Parallel tempering enabled:")
        print(f"  - Exchange interval: every {tempering_interval} steps")
        print(f"  - Exchange probability: {exchange_probability}")

    # Create results directory
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"five_agent_ppo_results_{experiment_time}"
    os.makedirs(results_dir, exist_ok=True)

    # Agent configurations with temperatures for parallel tempering
    agent_configs = [
        {"name": "Fresh Agent", "pretrained": None, "description": "No pre-training", "temperature": 1.0},
        {"name": "Bell-Trained Agent", "pretrained": ["bell"], "description": "Trained on Bell states only", "temperature": 1.2},
        {"name": "GHZ-Trained Agent", "pretrained": ["ghz"], "description": "Trained on GHZ states only", "temperature": 0.8},
        {"name": "W-Trained Agent", "pretrained": ["w"], "description": "Trained on W states only", "temperature": 0.8},
        {"name": "Multi-Trained Agent", "pretrained": ["ghz", "w"], "description": "Trained on three qubit states", "temperature": 0.6}
    ]
    
    # Sort agents by temperature for proper tempering (coldest to hottest)
    if enable_parallel_tempering:
        agent_configs.sort(key=lambda x: x["temperature"])
        print(f"\n🌡️ Temperature ladder (cold→hot):")
        for i, ac in enumerate(agent_configs):  # Changed variable name from 'config' to 'ac'
            print(f"  Agent {i}: {ac['name']} (T={ac['temperature']})")

    models, envs, scores = [], [], []
    training_curves = []
    exchange_history = []  # Track exchanges over time
    temperatures = [config["temperature"] for ac in agent_configs]

    print(f"\n🤖 Creating 5 agents:")
    for i, agent_config in enumerate(agent_configs):
        print(f"  Agent {i}: {agent_config['name']} - {agent_config['description']} (T={agent_config['temperature']})")
        
        # Create environment with temperature-based exploration
        env = DummyVecEnv([
            lambda agent_idx=i, global_cfg=config, temp=agent_config['temperature']: EnhancedMultiCircuitEnv(
                num_qubits=global_cfg.get("num_qubits", 3),  # Add fallback value
                max_steps=global_cfg.get("max_steps", 15),
                target_state=flat_target,
                temperature=temp,
                mode=global_cfg.get("mode", "custom"),
                reward_coeff=global_cfg.get("reward_coeff", 1.0),
                log_gate_history=log_gate_history,
                gate_history_logfile=(
                    f"{results_dir}/gate_history_agent{agent_idx}.jsonl"
                    if log_gate_history else None
                )
            )
        ])

         # Create model with temperature-dependent exploration
        ent_coef = 0.1 * agent_config['temperature']  # Higher temp = more exploration
        model = PPO(
            "MlpPolicy", env, device=device,
            ent_coef=ent_coef, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128])
        )

        # Load pretrained weights if specified
        if agent_config["pretrained"]:
            pretrained_models = []
            for state_type in agent_config["pretrained"]:
                if state_type in pretrained_model_paths:
                    pretrained_path = pretrained_model_paths[state_type]
                    pretrained_model = load_pretrained_model(
                        pretrained_path, env, device, target_qubits=config["num_qubits"]  # Now using the correct 'config'
                    )
                    if pretrained_model:
                        pretrained_models.append(pretrained_model)
            
            if pretrained_models:
                # Check if we need cross-architecture transfer
                needs_cross_arch = any(
                    detect_model_qubits(pretrained_model_paths[state_type]) != config["num_qubits"]
                    for state_type in agent_config["pretrained"]
                    if state_type in pretrained_model_paths
                )
                
                if needs_cross_arch:
                    success = transfer_weights_cross_architecture(pretrained_models, model, transfer_rate=0.8)
                    if success:
                        print(f"    ✅ Cross-architecture transfer from {agent_config['pretrained']} models")
                    else:
                        print(f"    ❌ Cross-architecture transfer failed for {agent_config['name']}")
                else:
                    # Standard transfer for same-architecture models
                    if len(pretrained_models) == 1:
                        success = transfer_weights_cross_architecture([pretrained_models[0]], model, transfer_rate=0.8)
                        print(f"    ✅ Transferred weights from {agent_config['pretrained'][0]} model")
                    else:
                        success = transfer_weights_cross_architecture(pretrained_models, model, transfer_rate=0.8)
                        print(f"    ✅ Transferred averaged weights from {agent_config['pretrained']} models")
            else:
                print(f"    ⚠️ Failed to load pretrained models for {agent_config['name']}")

        models.append(model)
        envs.append(env)

    # Live plotting setup
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))
        lines = [ax.plot([], [], label=agent_configs[i]["name"])[0] for i in range(5)]
        ax.set_xlabel("Training Steps (x1000)")
        ax.set_ylabel("Average Fidelity")
        ax.set_title("Live Training Progress - Five Agent Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show(block=False)

    # Training phase with parallel tempering
    print(f"\n🚀 Training all agents for {training_steps} steps")
    evaluation_interval = training_steps // 10  # Evaluate 10 times during training
    
    # Initialize exchange tracking
    total_exchanges = [0] * (len(models) - 1)
    exchange_attempts = [0] * (len(models) - 1)
    
    for step_phase in range(10):  # 10 evaluation phases
        current_step = step_phase * evaluation_interval
        print(f"\n📈 Training Phase {step_phase + 1}/10 (Steps {current_step}-{current_step + evaluation_interval})")
        
        # Train each agent for the interval
        for i, (model, env, agent_config) in enumerate(zip(models, envs, agent_configs)):
            print(f"  Training Agent {i}: {agent_config['name']} (T={agent_config['temperature']})...")
            
            # Break training into smaller chunks for tempering
            chunk_size = tempering_interval
            steps_remaining = evaluation_interval
            
            while steps_remaining > 0:
                train_steps = min(chunk_size, steps_remaining)
                model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
                steps_remaining -= train_steps
                
                # Perform parallel tempering exchange if enabled
                if enable_parallel_tempering and steps_remaining > 0:
                    # Quick evaluation for exchange decisions
                    for j, (m, e) in enumerate(zip(models, envs)):
                        quick_score = evaluate_agent_quick(m, e, episodes=2)
                        m._last_performance = quick_score
                    
                    # Attempt exchanges
                    exchanges = parallel_tempering_exchange(
                        models, temperatures, exchange_probability, verbose=True
                    )
                    
                    # Track exchange statistics
                    for j, exchange_made in enumerate(exchanges):
                        exchange_attempts[j] += 1
                        total_exchanges[j] += exchange_made
                        
                    if any(exchanges):
                        exchange_history.append({
                            'step': current_step + (evaluation_interval - steps_remaining),
                            'exchanges': exchanges.copy()
                        })
        
         # Evaluate all agents after this phase
        phase_scores = []
        for i, (model, env, agent_config) in enumerate(zip(models, envs, agent_configs)):
            fidelities = evaluate_agent(model, env, episodes=evaluation_episodes)
            avg_fidelity = np.mean(fidelities)
            phase_scores.append(avg_fidelity)
            
            if i >= len(training_curves):
                training_curves.append([])
            training_curves[i].append(avg_fidelity)
            
            print(f"    Agent {i} ({agent_config['name']}): {avg_fidelity:.4f}")
            
            # Update live plot
            if live_plot:
                x_data = [(j + 1) * evaluation_interval / 1000 for j in range(len(training_curves[i]))]
                lines[i].set_data(x_data, training_curves[i])
        
        if live_plot:
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.1)

    if live_plot:
        plt.ioff()
    
   # Print exchange statistics
    if enable_parallel_tempering:
        print(f"\n🔄 Parallel Tempering Statistics:")
        for i in range(len(total_exchanges)):
            if exchange_attempts[i] > 0:
                success_rate = total_exchanges[i] / exchange_attempts[i] * 100
                print(f"  Agent {i} ↔ Agent {i+1}: {total_exchanges[i]}/{exchange_attempts[i]} exchanges ({success_rate:.1f}%)")
            else:
                print(f"  Agent {i} ↔ Agent {i+1}: No exchange attempts")

    # Final evaluation
    print(f"\n📊 Final Evaluation")
    final_scores = []
    detailed_results = {
        "experiment_config": {
            "training_steps": training_steps,
            "evaluation_episodes": evaluation_episodes,
            "target_state": config["target_state"],
            "timestamp": experiment_time,
            "parallel_tempering_enabled": enable_parallel_tempering,
            "tempering_interval": tempering_interval if enable_parallel_tempering else None,
            "exchange_probability": exchange_probability if enable_parallel_tempering else None,
            "temperatures": temperatures if enable_parallel_tempering else None
        },
        "agents": [],
        "exchange_statistics": {
            "total_exchanges": total_exchanges if enable_parallel_tempering else [],
            "exchange_attempts": exchange_attempts if enable_parallel_tempering else [],
            "exchange_history": exchange_history if enable_parallel_tempering else []
        }
    }

    for i, (model, env, agent_config) in enumerate(zip(models, envs, agent_configs)):
        print(f"  Evaluating Agent {i}: {agent_config['name']}")
        
        # Comprehensive evaluation
        fidelities = evaluate_agent(model, env, episodes=50)  # More episodes for final eval
        
        agent_results = {
            "name": agent_config['name'],
            "description": agent_config['description'],
            "pretrained_models": agent_config['pretrained'],
            "final_avg_fidelity": float(np.mean(fidelities)),
            "final_max_fidelity": float(np.max(fidelities)),
            "final_std_fidelity": float(np.std(fidelities)),
            "training_curve": [float(x) for x in training_curves[i]],
            "all_final_fidelities": [float(x) for x in fidelities]
        }
        
        final_scores.append(agent_results["final_avg_fidelity"])
        detailed_results["agents"].append(agent_results)
        
        print(f"    Final Avg Fidelity: {agent_results['final_avg_fidelity']:.4f} ± {agent_results['final_std_fidelity']:.4f}")
        print(f"    Final Max Fidelity: {agent_results['final_max_fidelity']:.4f}")

    # Export trained models
    if models_export_dir is None:
        models_export_dir = f"{results_dir}/models"
    os.makedirs(models_export_dir, exist_ok=True)
    for i, model in enumerate(models):
        model_dir = os.path.join(models_export_dir, f"agent_{i}_{agent_configs[i]['name'].replace(' ', '_').lower()}")
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, "model.zip"))

    # Create comprehensive visualization with parallel tempering info
    visualize_results(
        detailed_results=detailed_results,
        agent_configs=agent_configs,
        training_curves=training_curves,
        exchange_history=exchange_history,
        enable_parallel_tempering=enable_parallel_tempering,
        results_dir=results_dir,
        evaluation_interval=evaluation_interval  # Add this parameter
    )

    # Create comprehensive visualization
    plt.figure(figsize=(20, 15))
    colors = ['gray', 'red', 'blue', 'green', 'purple']
    
    # 1. Training curves
    plt.subplot(2, 3, 1)
    for i, (curve, agent_config) in enumerate(zip(training_curves, agent_configs)):
        x_data = [(j + 1) * evaluation_interval / 1000 for j in range(len(curve))]
        plt.plot(x_data, curve, 'o-', color=colors[i], linewidth=2, markersize=6,
                label=agent_config['name'])
    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Average Fidelity")
    plt.title("Training Evolution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Final performance comparison
    plt.subplot(2, 3, 2)
    agent_names = [config['name'] for config in agent_configs]
    bars = plt.bar(range(5), final_scores, color=colors, alpha=0.7)
    plt.ylabel("Final Average Fidelity")
    plt.title("Final Performance Comparison")
    plt.xticks(range(5), [f"Agent {i}" for i in range(5)], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score, name in zip(bars, final_scores, agent_names):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Transfer learning effectiveness
    plt.subplot(2, 3, 3)
    fresh_score = final_scores[0]
    pretrained_scores = final_scores[1:4]  # Bell, GHZ, W
    multi_score = final_scores[4]
    
    transfer_improvements = [score - fresh_score for score in pretrained_scores]
    multi_improvement = multi_score - fresh_score
    
    labels = ['Bell', 'GHZ', 'W', 'Multi']
    improvements = transfer_improvements + [multi_improvement]
    bar_colors = colors[1:5]
    
    bars = plt.bar(labels, improvements, color=bar_colors, alpha=0.7)
    plt.ylabel("Fidelity Improvement over Fresh Agent")
    plt.title("Transfer Learning Benefits")
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005 if improvement >= 0 else bar.get_height() - 0.015,
                f'{improvement:.3f}', ha='center', va='bottom' if improvement >= 0 else 'top', fontweight='bold')
    
    # 4. Performance distribution
    plt.subplot(2, 3, 4)
    all_fidelities = [detailed_results["agents"][i]["all_final_fidelities"] for i in range(5)]
    plt.boxplot(all_fidelities, labels=[f"A{i}" for i in range(5)])
    plt.ylabel("Fidelity Distribution")
    plt.title("Performance Consistency")
    plt.grid(axis='y', alpha=0.3)
    
    # 5. Learning rate comparison
    plt.subplot(2, 3, 5)
    learning_rates = []
    for curve in training_curves:
        if len(curve) > 1:
            # Calculate average improvement per step
            total_improvement = curve[-1] - curve[0]
            steps = len(curve)
            learning_rates.append(total_improvement / steps)
        else:
            learning_rates.append(0)
    
    plt.bar(range(5), learning_rates, color=colors, alpha=0.7)
    plt.ylabel("Learning Rate (Fidelity/Step)")
    plt.title("Learning Speed Comparison")
    plt.xticks(range(5), [f"Agent {i}" for i in range(5)], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 6. Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""EXPERIMENT SUMMARY
    
Best Agent: {agent_configs[np.argmax(final_scores)]['name']}
Best Score: {max(final_scores):.4f}

Transfer Learning Results:
• Bell: {transfer_improvements[0]:+.3f}
• GHZ: {transfer_improvements[1]:+.3f}  
• W: {transfer_improvements[2]:+.3f}
• Multi: {multi_improvement:+.3f}

Key Findings:
• Multi-state training {'helps' if multi_improvement > max(transfer_improvements) else 'may not be optimal'}
• Best single-state: {['Bell', 'GHZ', 'W'][np.argmax(transfer_improvements)]}
• Transfer learning {'beneficial' if np.mean(transfer_improvements) > 0 else 'needs investigation'}
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/complete_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    with open(f"{results_dir}/detailed_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

     # Save training curves to CSV
    with open(f"{results_dir}/training_curves.csv", "w") as f:
        f.write("step,agent_0_fresh,agent_1_bell,agent_2_ghz,agent_3_w,agent_4_multi\n")
        for step_idx in range(len(training_curves[0])):
            step_num = (step_idx + 1) * evaluation_interval
            row = [str(step_num)]
            for agent_idx in range(5):
                if step_idx < len(training_curves[agent_idx]):
                    row.append(f"{training_curves[agent_idx][step_idx]:.6f}")
                else:
                    row.append("0.0")
            f.write(",".join(row) + "\n")

    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("FIVE-AGENT PPO COMPARISON EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    print(f"\n🏆 FINAL RANKINGS:")
    ranked_agents = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    for rank, (agent_idx, score) in enumerate(ranked_agents, 1):
        print(f"  {rank}. Agent {agent_idx} ({agent_configs[agent_idx]['name']}): {score:.4f}")
    
    fresh_idx = next(i for i, ac in enumerate(agent_configs) if ac['name'] == 'Fresh Agent')
    fresh_score = final_scores[fresh_idx]
    
    print(f"\n📈 TRANSFER LEARNING ANALYSIS:")
    print(f"  Fresh Agent Baseline: {fresh_score:.4f}")
    
    # Find indices for each pretrained agent
    bell_idx = next(i for i, ac in enumerate(agent_configs) if 'Bell' in ac['name'])
    ghz_idx = next(i for i, ac in enumerate(agent_configs) if 'GHZ' in ac['name'])
    w_idx = next(i for i, ac in enumerate(agent_configs) if 'W-' in ac['name'])
    multi_idx = next(i for i, ac in enumerate(agent_configs) if 'Multi' in ac['name'])
    
    transfer_improvements = [
        final_scores[bell_idx] - fresh_score,
        final_scores[ghz_idx] - fresh_score,
        final_scores[w_idx] - fresh_score
    ]
    multi_improvement = final_scores[multi_idx] - fresh_score
    
    for i, (state, improvement) in enumerate(zip(['Bell', 'GHZ', 'W'], transfer_improvements)):
        print(f"  {state} Transfer: {improvement:+.4f} ({improvement/fresh_score*100:+.1f}%)")
    print(f"  Multi-State Transfer: {multi_improvement:+.4f} ({multi_improvement/fresh_score*100:+.1f}%)")
    
    best_single = np.argmax(transfer_improvements)
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  • Best single-state training: {['Bell', 'GHZ', 'W'][best_single]}")
    print(f"  • Multi-state vs best single: {multi_improvement - transfer_improvements[best_single]:+.4f}")
    print(f"  • Transfer learning effectiveness: {np.mean(transfer_improvements):+.4f} average improvement")
    
    if multi_improvement > max(transfer_improvements):
        print(f"  • BREAKTHROUGH: Multi-state training outperforms single-state training!")
    else:
        print(f"  • Single-state training may be more effective than multi-state")
    
    if enable_parallel_tempering:
        total_exchange_rate = sum(total_exchanges) / max(sum(exchange_attempts), 1)
        print(f"\n🌡️ PARALLEL TEMPERING ANALYSIS:")
        print(f"  • Total successful exchanges: {sum(total_exchanges)}")
        print(f"  • Overall exchange success rate: {total_exchange_rate:.1%}")
        print(f"  • Temperature range: {min(temperatures):.1f} - {max(temperatures):.1f}")
        if total_exchange_rate > 0.1:
            print(f"  • Tempering appears beneficial for exploration-exploitation balance")
        else:
            print(f"  • Low exchange rate suggests temperature spacing may need adjustment")
    
    print(f"\n📁 Results saved to: {results_dir}/")
    return detailed_results

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
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

def get_default_config():
    """Get default configuration with all required parameters"""
    return {
        "num_qubits": 3,  # Number of qubits
        "max_steps": 15,  # Maximum steps per episode
        "reward_coeff": 1.0,  # Reward scaling coefficient
        "temperature": 1.0,  # Base temperature for exploration
        "mode": "custom",  # Environment mode
        "target_state": [  # Default target state for W state
            [0.40824829, 0.0], [-0.57735027, 0.0],
            [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
            [0.40824829, 0.0], [0.57735027, 0.0]
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-Agent PPO Comparison Experiment")
    parser.add_argument("--pretrained", nargs="*", default=[],
                        help="Pretrained model paths, e.g. --pretrained bell=path/to/bell.zip ghz=path/to/ghz.zip")
    parser.add_argument("--config", default="config.json", help="Config file (JSON)")
    parser.add_argument("--training-steps", type=int, default=50000, help="Total training steps per agent")
    parser.add_argument("--evaluation-episodes", type=int, default=10, help="Episodes for evaluation")
    parser.add_argument("--live-plot", action="store_true", help="Enable live plotting of training curves")
    parser.add_argument("--log-gate-history", action="store_true", help="Log full gate history per episode")
    parser.add_argument("--models-export-dir", type=str, default=None, help="Directory to export trained models")
    parser.add_argument("--enable-parallel-tempering", action="store_true", default=True, 
                        help="Enable parallel tempering between agents")
    parser.add_argument("--disable-parallel-tempering", action="store_true", 
                        help="Disable parallel tempering")
    parser.add_argument("--tempering-interval", type=int, default=5000, 
                        help="Steps between tempering attempts")
    parser.add_argument("--exchange-probability", type=float, default=0.1, 
                        help="Probability of attempting exchange between adjacent agents")
    args = parser.parse_args()

    # Handle parallel tempering flag logic
    if args.disable_parallel_tempering:
        enable_tempering = False
    else:
        enable_tempering = args.enable_parallel_tempering

    # Load and validate config
    config = get_default_config()  # Start with defaults
    if os.path.exists(args.config):
        loaded_config = load_config(args.config)
        # Update default config with loaded values
        config.update(loaded_config)
    
    # Validate required parameters
    required_params = ["num_qubits", "max_steps", "mode"]
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        print(f"❌ Missing required parameters in config: {', '.join(missing_params)}")
        print(f"Using defaults: {', '.join(f'{param}={config[param]}' for param in missing_params)}")

    # Parse pretrained models
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
    missing_models = []
    for name, path in pretrained_paths.items():
        if os.path.exists(path):
            valid_paths[name] = path
            print(f"✅ Found {name} model: {path}")
        else:
            missing_models.append(f"{name}={path}")
    
    if not valid_paths:
        print("❌ No pre-trained models found!")
        print("Please provide with --pretrained bell=path.zip ghz=path.zip w=path.zip")
        print("Expected files:")
        for name in ["bell", "ghz", "w"]:
            print(f"  {name}: ppo_{name}_agent_finetuned.zip")
        exit(1)
    
    if missing_models:
        print(f"⚠️ Missing models: {', '.join(missing_models)}")
        print("Experiment will continue with available models only.")

    # Run the experiment
    results = run_five_agent_experiment(
        valid_paths,
        config,
        training_steps=args.training_steps,
        evaluation_episodes=args.evaluation_episodes,
        live_plot=args.live_plot,
        log_gate_history=args.log_gate_history,
        models_export_dir=args.models_export_dir,
        enable_parallel_tempering=enable_tempering,
        tempering_interval=args.tempering_interval,
        exchange_probability=args.exchange_probability
    )