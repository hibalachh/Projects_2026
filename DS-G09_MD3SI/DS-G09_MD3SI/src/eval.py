"""Evaluation, figures, and before/after videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd

from .dqn import DQNAgent
from .env import RansomwareEnv
from .utils import ensure_dirs, load_config, set_seed


def make_env(cfg):
    e = cfg["environment"]
    return RansomwareEnv(
        n_pc=e["n_pc"], max_steps=e["max_steps"], infection_probability=e["infection_probability"],
        reward_new_infection=e["reward_new_infection"], reward_action_cost=e["reward_action_cost"],
        reward_contained=e["reward_contained"], reward_total_compromise=e["reward_total_compromise"],
    )


def evaluate_seed(cfg, agent, seed):
    env = make_env(cfg)
    rows = []
    set_seed(seed)
    for episode in range(1, int(cfg["evaluation"]["episodes"]) + 1):
        obs, _ = env.reset(seed=10000 + seed * 1000 + episode)
        total = 0.0
        max_infected = 0
        steps = 0
        while True:
            action = agent.choose_action(obs.astype(np.float32), explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            max_infected = max(max_infected, info["nb_infectes"])
            if terminated or truncated:
                rows.append({
                    "seed": seed,
                    "episode": episode,
                    "total_reward": total,
                    "final_infected": info["nb_infectes"],
                    "max_infected": max_infected,
                    "isolated": info["isolated"],
                    "patched": info["patched"],
                    "steps": steps,
                })
                break
    env.close()
    return rows


def save_learning_curve(training_csv, output_pdf, window):
    df = pd.read_csv(training_csv)
    ep = df.groupby(["seed", "episode"], as_index=False).agg(episode_return=("episode_return", "last"))
    pivot = ep.pivot(index="episode", columns="seed", values="episode_return").sort_index()
    smooth = pivot.rolling(window, min_periods=1).mean()
    mean = smooth.mean(axis=1)
    std = smooth.std(axis=1).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(mean.index, mean.values, label="Moyenne")
    ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.2, label="± écart-type")
    ax.set_title("Courbe d'apprentissage DQN")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Retour cumulé")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def save_evaluation_figure(eval_csv, output_pdf):
    df = pd.read_csv(eval_csv)
    metrics = {
        "Reward moyen": df["total_reward"].mean(),
        "Infections finales": df["final_infected"].mean(),
        "Pic d'infections": df["max_infected"].mean(),
    }
    means = list(metrics.values())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(list(metrics.keys()), means)
    ax.set_title("Performance moyenne après entraînement")
    ax.set_ylabel("Valeur moyenne")
    ax.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def capture_video(cfg, agent, output_path, seed, trained):
    env = make_env(cfg)
    obs, info = env.reset(seed=seed)
    frames = []
    total = 0.0
    labels = {0: "SAIN", 1: "INFECTE", 2: "ISOLE", 3: "PATCHE"}
    for _ in range(int(cfg["environment"]["max_steps"])):
        action = agent.choose_action(obs.astype(np.float32), explore=not trained)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        frames.append((env.current_step, env.network_state.copy(), action, total))
        if terminated or truncated:
            break
    env.close()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    writer = FFMpegWriter(fps=2, metadata={"title": "Ransomware containment"})
    with writer.saving(fig, output_path, dpi=130):
        for step, states, action, total_reward in frames:
            ax.clear()
            x = np.arange(len(states))
            values = states.astype(float)
            ax.bar(x, values)
            ax.set_ylim(-0.2, 3.4)
            ax.set_xticks(x, [f"PC{i+1}" for i in x])
            ax.set_yticks([0, 1, 2, 3], [labels[i] for i in range(4)])
            ax.set_title(f"{'Après entraînement' if trained else 'Avant entraînement'} - étape {step}")
            ax.set_ylabel("État du PC")
            ax.text(0.99, 1.02, f"Action={action} | Reward cumulé={total_reward:.1f}", transform=ax.transAxes, ha="right")
            ax.grid(axis="y", alpha=0.15)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    env = make_env(cfg)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg["training"])
    agent.load(cfg["paths"]["model"])
    env.close()

    all_rows = []
    for seed in cfg["experiment"]["seeds"]:
        print(f"=== Evaluation seed {seed} ===")
        all_rows.extend(evaluate_seed(cfg, agent, int(seed)))
    ev = pd.DataFrame(all_rows)
    ev.to_csv(cfg["paths"]["evaluation_csv"], index=False)
    save_learning_curve(cfg["paths"]["training_csv"], cfg["paths"]["learning_curve"], int(cfg["experiment"]["window"]))
    save_evaluation_figure(cfg["paths"]["evaluation_csv"], cfg["paths"]["evaluation_figure"])

    # Before training: a fresh, untrained DQN (epsilon-greedy exploration).
    fresh = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg["training"])
    capture_video(cfg, fresh, cfg["paths"]["before_video"], 123, trained=False)
    # After training: the learned policy used above, with exploration disabled.
    capture_video(cfg, agent, cfg["paths"]["after_video"], 123, trained=True)
    print(f"Saved {cfg['paths']['evaluation_csv']}")
    print(f"Saved {cfg['paths']['learning_curve']}")
    print(f"Saved {cfg['paths']['evaluation_figure']}")
    print(f"Saved {cfg['paths']['before_video']}")
    print(f"Saved {cfg['paths']['after_video']}")


if __name__ == "__main__":
    main()
