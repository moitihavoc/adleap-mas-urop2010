import os
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.communication.traces import TRACE_FIELDS
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.coop_expert import coop_expert_action
from src.utils.evaluation import ADHOC_ANGLE, ADHOC_LEVEL, ADHOC_RADIUS, DIM, NAGENTS, NTASKS
from src.utils.scenarios import make_scenario

'''
=== Figures for the Gate 0 notes ===
1. val_split_exercise.png: training vs. validation accuracy per epoch, read from the saved output of the
   validation-split exercise (results_analysis/Results/val_split_exercise.txt).
2. trace_example.png: the 6 intent trace channels (full field, before local reading) after 8 teacher steps
   of one hard scenario, with agent and task positions marked.
'''

SURFACE, INK, MUTED = '#fcfcfb', '#222222', '#777777'
BLUE, ORANGE = '#2a78d6', '#eb6834'  # dataviz categorical slots 1-2 (validated)


def plot_val_split(src="results_analysis/Results/val_split_exercise.txt",
                   out="results_analysis/Plots/val_split_exercise.png"):
    rows = re.findall(r"Epoch (\d+)/\d+ - training accuracy: ([0-9.]+), validation accuracy: ([0-9.]+)",
                      open(src).read())
    epochs = [int(r[0]) for r in rows]
    train = [float(r[1]) for r in rows]
    val = [float(r[2]) for r in rows]
    best = int(np.argmax(val))

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(epochs, train, color=BLUE, linewidth=2, marker='o', markersize=4, label='Training games (240)')
    ax.plot(epochs, val, color=ORANGE, linewidth=2, marker='o', markersize=4, label='Validation games (60)')
    ax.annotate(f"best validation {val[best]:.3f} (epoch {epochs[best]})", (epochs[best], val[best]),
                xytext=(epochs[best] + 2, val[best] - 0.12), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=1))
    ax.text(epochs[-1], train[-1] + 0.015, f"{train[-1]:.3f}", fontsize=8, color=INK, ha='right')
    ax.text(epochs[-1], val[-1] - 0.05, f"{val[-1]:.3f}", fontsize=8, color=INK, ha='right')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy vs. teacher (L1)')
    ax.set_ylim(0.3, 1.02)
    ax.set_title('Validation-split exercise: the gap that shows overfitting (old data)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_trace_example(out="results_analysis/Plots/trace_example.png", seed=7, steps=8):
    random.seed(seed)
    np.random.seed(seed)
    base_env = make_scenario(dim=DIM, nagents=NAGENTS, ntasks=NTASKS, adhoc_level=ADHOC_LEVEL,
                             adhoc_radius=ADHOC_RADIUS, adhoc_angle=ADHOC_ANGLE, task_level_range=(0.3, 1.5))
    env = StigmergicLevelForagingEnv(base_env, dim=DIM, trace_mode="intent")
    env.reset()
    for _ in range(steps):
        env.step(coop_expert_action(env.env))

    fields = env.traces.fields
    names = TRACE_FIELDS["intent"]
    ranges = {'bearing_x': (-1, 1), 'bearing_y': (-1, 1)}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), facecolor=SURFACE)
    for ax, name, field in zip(axes.flat, names, fields):
        low, high = ranges.get(name, (0, 1))
        cmap = 'RdBu_r' if name.startswith('bearing') else 'Blues'
        image = ax.imshow(field.T, origin='lower', cmap=cmap, vmin=low, vmax=high)
        for (x, y), value in np.ndenumerate(field):
            if value != 0:
                ax.text(x, y, f"{value:.1f}", ha='center', va='center', fontsize=6,
                        color='white' if abs(value) > 0.6 * max(abs(low), high) else INK)
        for agent in env.env.components['agents']:
            x, y = agent.position
            label = 'A' if agent.index == env.env.get_adhoc_agent().index else agent.index
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=2))
            ax.text(x - 0.45, y + 0.2, label, fontsize=6, color=ORANGE, fontweight='bold')
        for task in env.env.components['tasks']:
            if not task.completed:
                x, y = task.position
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=INK,
                                           linewidth=1, linestyle='--'))
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xticks(range(DIM))
        ax.set_yticks(range(DIM))
        ax.tick_params(labelsize=6, colors=MUTED)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    fig.suptitle(f"Intent traces after {steps} steps (orange boxes: agents, A = ad hoc; dashed: unfinished tasks)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs("results_analysis/Plots", exist_ok=True)
    plot_val_split()
    plot_trace_example()
