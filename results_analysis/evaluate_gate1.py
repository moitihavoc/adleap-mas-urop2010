import argparse
import csv
import os
import random
import subprocess
import sys

import numpy as np
import torch

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from results_analysis.gate1_learning_curve import NUM_HELDOUT_EPISODES, fit, load_data
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.models.cnn_policy import TraceCNN
from src.reasoning.levelbased.l1 import l1_planning
from src.utils.coop_expert import coop_expert_action
from src.utils.evaluation import DIM, create_evaluation_set

'''
=== Closed-loop evaluation of the Gate 1 student against the baselines ===
The student is trained on the whole Gate 1 training pool (2,000 teacher episodes, 15-channel input) with early stopping
on the 100 held-out episodes, then plays the 100 hard evaluation scenarios (seed 2026, task levels U(0.3, 1.5)).
Every episode seeds random / np.random with its scenario index, exactly as results_analysis/evaluate_pomcp.py on
post_urop1010 did, so idle / l1 / teacher here must reproduce the logged 21 Oct numbers (a check of the ported code).
POMCP rows are not rerun (about 30 min each): their per-episode CSVs are read from post_urop1010.

Usage:
    python3 results_analysis/evaluate_gate1.py --train-student
    python3 results_analysis/evaluate_gate1.py --evaluate
    python3 results_analysis/evaluate_gate1.py --plot
'''

TASK_LEVELS = (0.3, 1.5)
MAX_STEPS = 200
WEIGHTS = "src/models/gate1_student_intent_tasks0.3-1.5.pth"
OUT_DIR = "results_analysis/Results/gate1_eval"
POMCP_BRANCH_DIR = "results_analysis/Results/pomcp_intent_tasks0.3-1.5"
PLOT_PATH = "results_analysis/Plots/gate1_vs_baselines.png"
CONFIGS = ("idle", "random", "l1", "teacher", "student-argmax", "student-sample")
POMCP_CONFIGS = ("pomcp-view-0.5", "pomcp-full-0.5")


def train_student():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    episode_ids = sorted(set(data['episode'].tolist()))
    random.Random(2026).shuffle(episode_ids)                     # same held-out split as the learning curve
    heldout = torch.tensor(episode_ids[:NUM_HELDOUT_EPISODES])
    val_mask = torch.isin(data['episode'], heldout)
    train_mask = ~val_mask
    best = fit(data['obs'][train_mask].float().to(device), data['action'][train_mask].to(device),
               data['obs'][val_mask].float().to(device), data['action'][val_mask].to(device),
               seed=0, device=device, return_model=True)
    torch.save(best['model'].state_dict(), WEIGHTS)
    print(f"student: {int(train_mask.sum())} training states, held-out accuracy {best['val_acc']:.3f} "
          f"(epoch {best['epoch']}) -> {WEIGHTS}")


def make_policy(name, model):
    def choose(env, obs, scenario_index):
        adhoc = env.env.get_adhoc_agent()
        if name == "idle":
            return 4
        if name == "random":
            return random.randrange(5)
        if name == "l1":
            action, target = l1_planning(env.env.observation_space(env.env.copy()), adhoc)
            adhoc.target = target
            return action
        if name == "teacher":
            return coop_expert_action(env.env)
        with torch.no_grad():
            logits, _ = model(obs.unsqueeze(0))
        if name == "student-argmax":
            return int(logits.argmax(dim=1).item())
        return int(torch.multinomial(torch.softmax(logits[0], dim=0), 1).item())   # student-sample
    return choose


def evaluate():
    scenarios = create_evaluation_set(100, task_level_range=TASK_LEVELS)
    model = TraceCNN(input_channels=15, num_actions=5)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    for name in CONFIGS:
        choose = make_policy(name, model)
        rows = []
        for i, scenario in enumerate(scenarios):
            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)
            env = StigmergicLevelForagingEnv(scenario.copy(), dim=DIM, trace_mode="intent", self_features=True)
            obs = env.reset()
            ntasks = len(env.env.components['tasks'])
            reward, step, done = 0, 0, False
            while not done and step < MAX_STEPS:
                obs, r, done, _ = env.step(choose(env, obs, i))
                reward += r
                step += 1
            rows.append({'scenario': i, 'reward': reward, 'steps': step, 'completed': int(reward >= ntasks),
                         'timeout': int(step >= MAX_STEPS and not done)})
        with open(f"{OUT_DIR}/{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        completed = sum(r['completed'] for r in rows)
        print(f"{name:15s} all tasks {completed:3d}% | mean reward {np.mean([r['reward'] for r in rows]):.2f} "
              f"| mean steps {np.mean([r['steps'] for r in rows]):.1f}")


def wilson(successes, n, z=1.96):
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def read_config(name):
    if name in POMCP_CONFIGS:
        text = subprocess.run(["git", "show", f"post_urop1010:{POMCP_BRANCH_DIR}/{name}.csv"],
                              capture_output=True, text=True, check=True).stdout
        rows = list(csv.DictReader(text.splitlines()))
    else:
        with open(f"{OUT_DIR}/{name}.csv") as f:
            rows = list(csv.DictReader(f))
    return [int(r['completed']) for r in rows], [int(r['steps']) for r in rows]


def plot():
    import matplotlib.pyplot as plt

    labels = {
        "idle": "Idle (never moves)", "random": "Random moves", "l1": "L1 heuristic",
        "student-argmax": "My student, top move", "student-sample": "My student, sampled move",
        "teacher": "Cooperative teacher", "pomcp-view-0.5": "POMCP, own view, 0.5 s*",
        "pomcp-full-0.5": "POMCP, full info, 0.5 s*",
    }
    order = ["idle", "random", "l1", "student-argmax", "student-sample", "teacher", "pomcp-view-0.5", "pomcp-full-0.5"]
    stats = []
    for name in order:
        completed, steps = read_config(name)
        n = len(completed)
        low, high = wilson(sum(completed), n)
        stats.append((name, sum(completed) / n, low, high, np.mean(steps), 1.96 * np.std(steps, ddof=1) / np.sqrt(n)))

    surface, ink, muted, grey, blue = '#fcfcfb', '#222222', '#777777', '#a7a6a0', '#2a78d6'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=surface, sharey=True)
    y = np.arange(len(stats))
    colors = [blue if name.startswith("student") else grey for name, *_ in stats]
    for ax, values, errors, title, xlabel in (
            (ax1, [s[1] * 100 for s in stats], [[max(s[1] - s[2], 0) * 100 for s in stats], [max(s[3] - s[1], 0) * 100 for s in stats]],
             "All tasks finished (higher is better)", "% of 100 hard scenarios, 95% CI"),
            (ax2, [s[4] for s in stats], [s[5] for s in stats],
             "Steps per episode (lower is better)", "mean steps, 95% CI (200 = timed out)")):
        ax.set_facecolor(surface)
        ax.barh(y, values, color=colors, height=0.62, edgecolor=surface, linewidth=2)
        ax.errorbar(values, y, xerr=errors, fmt='none', ecolor=ink, elinewidth=1, capsize=3)
        uppers = errors[1] if ax is ax1 else errors
        for yi, v, upper in zip(y, values, uppers):
            ax.text(v + upper + (1.5 if ax is ax1 else 3), yi, f"{v:.0f}{'%' if ax is ax1 else ''}", va='center',
                    fontsize=8, color=ink)
        ax.set_title(title, fontsize=10, color=ink)
        ax.set_xlabel(xlabel, fontsize=8, color=muted)
        ax.grid(axis='x', alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    ax1.set_xlim(0, 115)
    ax2.set_xlim(0, 210)
    ax1.set_yticks(y, [labels[s[0]] for s in stats], fontsize=8)
    ax1.invert_yaxis()
    fig.suptitle("Gate 1 student in real games vs. baselines (hard tasks, intent traces)", fontsize=11, color=ink)
    fig.text(0.01, 0.01, "* POMCP rows: logged 21 Oct run on post_urop1010, same scenarios and per-episode seeds; "
             "not rerun.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"saved {PLOT_PATH}")
    for name, rate, low, high, steps, steps_ci in stats:
        print(f"{name:15s} {rate * 100:5.1f}% [{low * 100:.0f}, {high * 100:.0f}]  steps {steps:6.1f} ± {steps_ci:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 1 student vs. baselines in closed-loop games.")
    parser.add_argument("--train-student", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.train_student:
        train_student()
    if args.evaluate:
        evaluate()
    if args.plot:
        plot()
