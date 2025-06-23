import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ppo_with_identity import MultiCircuitEnv, swap_models, evaluate_model_performance, calculate_swap_probability, smooth_curve

def auto_tune_coeffs(perf_scores: List[float], current_coeff: float) -> float:
    """
    Adjust reward coefficient based on validation improvements.
    Penalize if plateauing; increase if improving.
    """
    if len(perf_scores) < 3:
        return current_coeff

    if perf_scores[-1] > perf_scores[-2] + 0.01:
        return current_coeff * 1.05
    elif perf_scores[-1] < perf_scores[-2] - 0.01:
        return current_coeff * 0.95
    return current_coeff

def main():
    with open("parallel_tempering_config.json") as f:
        cfg = json.load(f)

    num_agents = len(cfg["temperatures"])
    flat_target = np.array([complex(r, i) for r, i in cfg["target_state"]], dtype=complex)
    flat_target /= np.linalg.norm(flat_target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, envs, scores = [], [], [[] for _ in range(num_agents)]
    reward_coeffs = [1.0 for _ in range(num_agents)]

    for i, T in enumerate(cfg["temperatures"]):
        env = DummyVecEnv([lambda T=T: MultiCircuitEnv(
            num_qubits=cfg["num_qubits"],
            max_steps=cfg["max_steps"],
            target_state=flat_target,
            temperature=T,
            reward_coeff=reward_coeffs[i]
        )])
        model = PPO("MlpPolicy", env, device=device, verbose=0)
        models.append(model)
        envs.append(env)

    swap_every = cfg["swap_interval"]
    steps_per_agent = cfg["timesteps_per_agent"]
    eval_every = cfg.get("evaluation_interval", swap_every * 2)

    for step in range(0, steps_per_agent, swap_every):
        current_scores = []
        for i in range(num_agents):
            models[i].learn(total_timesteps=swap_every, reset_num_timesteps=False)
            mean_fid, _ = evaluate_model_performance(models[i], envs[i], num_episodes=5)
            scores[i].append(mean_fid)
            current_scores.append(mean_fid)
            reward_coeffs[i] = auto_tune_coeffs(scores[i], reward_coeffs[i])
            envs[i].envs[0].reward_coeff = reward_coeffs[i]

        # Swap models based on PT
        for i in range(num_agents - 1):
            p = calculate_swap_probability(current_scores[i], current_scores[i+1], cfg["temperatures"][i], cfg["temperatures"][i+1])
            if np.random.rand() < p:
                swap_models(models[i], models[i+1])
                print(f"Swapped models {i} and {i+1} (p={p:.2f})")

    os.makedirs("pt_logs", exist_ok=True)
    for i, score_list in enumerate(scores):
        plt.plot(smooth_curve(score_list), label=f"T={cfg['temperatures'][i]}")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Fidelity")
    plt.title("Fidelity per Agent")
    plt.savefig("pt_logs/fidelity_vs_round.png")
    plt.close()

if __name__ == "__main__":
    main()