import argparse
import csv
import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.models.cnn_policy import TraceCNN
from src.utils.answer_key import answer_key_planes
from src.utils.coop_expert import coop_expert_action
from src.utils.evaluation import ADHOC_ANGLE, ADHOC_LEVEL, ADHOC_RADIUS, DIM, NAGENTS, NTASKS
from src.utils.scenarios import make_scenario
from src.utils.train_cnn import accuracy

'''
=== GATE 1: can the student learn the teacher's rule? (learning curve with a positive control) ===
Collect: the cooperative teacher drives on the hard task set (U(0.3, 1.5)), with 20% random actions so the data also
covers states off the teacher's path; every step is labelled with the teacher's action. Each step stores the student
input (15 channels: intent traces, visible levels, own position and heading) and the answer-key planes.
Train: held-out accuracy (split by episode) for growing training sets, two inputs:
    student            the 15-channel student input
    student+answer     the same plus the 3 answer-key planes (all hidden information the teacher uses)
Pass mark: >= 90% held-out accuracy for student+answer.

Usage:
    python3 results_analysis/gate1_learning_curve.py --collect --episodes 200 --worker 0
    python3 results_analysis/gate1_learning_curve.py --train --sizes 25 50 100 200 400 --seeds 0 1
'''

TASK_LEVELS = (0.3, 1.5)
MAX_STEPS = 200
RANDOM_ACTION_RATE = 0.2
DATA_GLOB = "src/Training_Data/gate1_intent_tasks0.3-1.5_w*.pt"
RESULTS_CSV = "results_analysis/Results/gate1_learning_curve.csv"
PLOT_PATH = "results_analysis/Plots/gate1_learning_curve.png"
NUM_HELDOUT_EPISODES = 100


def collect(num_episodes, worker):
    """Teacher-driven episodes (with random actions) labelled with the teacher's action; saved per worker."""
    obs_list, key_list, action_list, episode_list = [], [], [], []
    rng = random.Random(10_000 + worker)
    timeouts, start = 0, time.time()
    for ep in range(num_episodes):
        episode_id = worker * 100_000 + ep
        random.seed(episode_id)
        np.random.seed(episode_id)
        base_env = make_scenario(dim=DIM, nagents=NAGENTS, ntasks=NTASKS, adhoc_level=ADHOC_LEVEL,
                                 adhoc_radius=ADHOC_RADIUS, adhoc_angle=ADHOC_ANGLE, task_level_range=TASK_LEVELS)
        env = StigmergicLevelForagingEnv(base_env, dim=DIM, trace_mode="intent", self_features=True)
        obs = env.reset()

        done, step = False, 0
        while not done and step < MAX_STEPS:
            teacher_action = coop_expert_action(env.env)
            obs_list.append(obs.half())
            key_list.append(answer_key_planes(env.env, DIM).half())
            action_list.append(teacher_action)
            episode_list.append(episode_id)

            action = rng.randrange(5) if rng.random() < RANDOM_ACTION_RATE else teacher_action
            obs, _, done, _ = env.step(action)
            step += 1
        timeouts += not done
        if (ep + 1) % 10 == 0:
            print(f"[worker {worker}] {ep + 1}/{num_episodes} episodes, {len(action_list)} states, "
                  f"{time.time() - start:.0f} s", flush=True)

    path = DATA_GLOB.replace("*", str(worker))
    torch.save({'obs': torch.stack(obs_list), 'answer_key': torch.stack(key_list),
                'action': torch.tensor(action_list), 'episode': torch.tensor(episode_list),
                'timeouts': timeouts, 'episodes': num_episodes}, path)
    print(f"[worker {worker}] saved {len(action_list)} states from {num_episodes} episodes "
          f"({timeouts} timed out) to {path}")


def load_data():
    parts = [torch.load(path) for path in sorted(glob.glob(DATA_GLOB))]
    data = {key: torch.cat([part[key] for part in parts]) for key in ('obs', 'answer_key', 'action', 'episode')}
    data['timeouts'] = sum(part['timeouts'] for part in parts)
    data['episodes'] = sum(part['episodes'] for part in parts)
    return data


def chunked_accuracy(model, obs, actions, chunk=4096):
    """accuracy() from train_cnn.py, applied chunk by chunk so a large set fits in GPU memory."""
    total = sum(accuracy(model, obs[i:i + chunk], actions[i:i + chunk]) * len(actions[i:i + chunk])
                for i in range(0, len(actions), chunk))
    return total / len(actions)


def fit(train_x, train_y, val_x, val_y, seed, device, max_epochs=40, patience=5, batch_size=128, return_model=False):
    """Policy-only behaviour cloning with early stopping on validation accuracy; returns the best epoch's numbers
    (and, with return_model, the model restored to that epoch's weights, on the CPU)."""
    torch.manual_seed(seed)
    model = TraceCNN(input_channels=train_x.shape[1], num_actions=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best = {'val_acc': -1.0}
    epochs_without_gain = 0
    for epoch in range(max_epochs):
        indices = torch.randperm(train_x.shape[0], device=device)
        for i in range(0, train_x.shape[0], batch_size):
            batch = indices[i:i + batch_size]
            pi_logits, _ = model(train_x[batch])
            loss = criterion(pi_logits, train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = chunked_accuracy(model, val_x, val_y)
        if val_acc > best['val_acc']:
            best = {'val_acc': val_acc, 'train_acc': chunked_accuracy(model, train_x, train_y), 'epoch': epoch + 1}
            if return_model:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break
    if return_model:
        model = model.cpu()
        model.load_state_dict(best_state)
        best['model'] = model.eval()
    return best


def train(sizes, seeds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    episode_ids = sorted(set(data['episode'].tolist()))
    random.Random(2026).shuffle(episode_ids)
    heldout, pool = episode_ids[:NUM_HELDOUT_EPISODES], episode_ids[NUM_HELDOUT_EPISODES:]
    print(f"{data['episodes']} episodes ({data['timeouts']} timed out), {len(data['action'])} states | "
          f"held out {len(heldout)} episodes, training pool {len(pool)} episodes | device {device}")

    def select(ids):
        mask = torch.isin(data['episode'], torch.tensor(ids))
        return mask

    val_mask = select(heldout)
    val_y = data['action'][val_mask].to(device)
    majority = torch.bincount(val_y, minlength=5).max().item() / len(val_y)
    print(f"held-out states: {len(val_y)} | majority-class accuracy {majority:.3f}")

    inputs = {
        'student': lambda mask: data['obs'][mask].float(),
        'student+answer': lambda mask: torch.cat([data['obs'][mask], data['answer_key'][mask]], dim=1).float(),
    }

    rows = []
    for size in sizes:
        if size > len(pool):
            print(f"skip size {size}: only {len(pool)} training episodes")
            continue
        for seed in seeds:
            train_ids = random.Random(seed).sample(pool, size)
            assert not set(train_ids) & set(heldout)
            train_mask = select(train_ids)
            for name, build in inputs.items():
                train_x, val_x = build(train_mask).to(device), build(val_mask).to(device)
                train_y = data['action'][train_mask].to(device)
                start = time.time()
                best = fit(train_x, train_y, val_x, val_y, seed, device)
                row = {'input': name, 'train_episodes': size, 'train_states': len(train_y), 'seed': seed,
                       'train_acc': round(best['train_acc'], 4), 'heldout_acc': round(best['val_acc'], 4),
                       'best_epoch': best['epoch'], 'majority_acc': round(majority, 4)}
                rows.append(row)
                print(f"{name:15s} episodes={size:4d} states={len(train_y):6d} seed={seed} "
                      f"train={best['train_acc']:.3f} held-out={best['val_acc']:.3f} "
                      f"(epoch {best['epoch']}, {time.time() - start:.0f} s)", flush=True)
                del train_x, val_x
                write_csv(rows)

    print(f"saved {RESULTS_CSV}")
    plot(rows)


def write_csv(rows):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows=None):
    import matplotlib.pyplot as plt

    if rows is None:
        with open(RESULTS_CSV) as f:
            rows = [{k: (float(v) if k not in ('input',) else v) for k, v in r.items()} for r in csv.DictReader(f)]

    colors = {'student': '#2a78d6', 'student+answer': '#eb6834'}  # dataviz categorical slots 1-2, validated
    labels = {'student': 'Student input (15 ch)', 'student+answer': 'Student + answer key (18 ch)'}
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for name, color in colors.items():
        own = [r for r in rows if r['input'] == name]
        states = sorted({r['train_states'] for r in own})
        sizes = sorted({r['train_episodes'] for r in own})
        means, xs = [], []
        for size in sizes:
            group = [r for r in own if r['train_episodes'] == size]
            xs.append(np.mean([r['train_states'] for r in group]))
            means.append(np.mean([r['heldout_acc'] for r in group]))
            ax.scatter([r['train_states'] for r in group], [r['heldout_acc'] for r in group],
                       color=color, alpha=0.35, s=18)
        ax.plot(xs, means, marker='o', markersize=6, linewidth=2, color=color, label=labels[name] + " held-out")
        train_means = [np.mean([r['train_acc'] for r in own if r['train_episodes'] == s]) for s in sizes]
        ax.plot(xs, train_means, linestyle=':', linewidth=2, color=color, label=labels[name] + " train")
    ax.axhline(0.9, color='#444444', linestyle='--', linewidth=1)
    ax.text(0.01, 0.905, 'Phase 1 pass mark (90%)', fontsize=8, color='#444444', va='bottom', transform=ax.get_yaxis_transform())
    ax.axhline(rows[0]['majority_acc'], color='#999999', linestyle=':', linewidth=1)
    ax.text(0.01, rows[0]['majority_acc'] + 0.005, 'always the most common action', fontsize=8,
            color='#777777', va='bottom', transform=ax.get_yaxis_transform())
    ax.set_xscale('log')
    ax.set_xlabel('Training states (log scale)')
    ax.set_ylabel('Accuracy vs. teacher')
    ax.set_ylim(0, 1.02)
    ax.set_title('Phase 1 learning curve: cooperative teacher, hard tasks, intent traces')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"saved {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 1 learning curve with an answer-key positive control.")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    if args.collect:
        collect(args.episodes, args.worker)
    if args.train:
        train(args.sizes, args.seeds)
    if args.plot:
        plot()
