import argparse
import csv
import glob
import math
import os
import random
import sys
import time

import numpy as np

sys.path.append(os.getcwd())

from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.envs.TracePlanningEnv import TracePlanningEnv
from src.utils.coop_expert import coop_expert_action
from src.utils.evaluation import DIM, create_evaluation_set, run_pomcp_episode

'''
=== Does POMCP play better when its belief reads the traces? (C0 / C1 / C2) ===
Method and reasoning: docs/trace_pomcp/trace_pomcp_method.tex (Sections 5-7).

Two measurements on the 100 shared hard evaluation scenarios (create_evaluation_set, seed 2026, task levels U(0.3, 1.5);
intent traces; the ad hoc agent has its own view, radius 0.25):

1. --belief: belief accuracy (no planning). The ad hoc agent is driven by the cooperative teacher (only to visit
   realistic states; it is not part of any belief). At each of the first BELIEF_STEPS steps, C1 and C2 each draw
   PARTICLES guesses from the same observation, and we measure, for every hidden teammate and every hidden task, how far
   the guesses are from the truth. Writes Results/trace_pomcp/belief_accuracy.csv (one row per hidden object per step).

2. --games: POMCP plays as the ad hoc agent. Configurations (name: belief, tree reuse, seconds per decision):
       C0-<b>: framework world (no traces, hidden teammates dropped), tree reuse (the October set-up)
       C1-<b>: TracePlanningEnv, belief "uniform" (corrected, no traces), new tree every decision
       C2-<b>: TracePlanningEnv, belief "traces"  (reads traces),          new tree every decision
   One CSV per configuration and worker in Results/trace_pomcp/games/. Scenarios already on file are skipped (resume).
   Every episode seeds random / np.random with its scenario index (as evaluate_gate1.py).

3. --summary: tables for both measurements (completion with Wilson 95% CI, steps, paired C2 vs C1).

Usage:
    python3 results_analysis/evaluate_trace_pomcp.py --belief
    for w in $(seq 0 17); do python3 results_analysis/evaluate_trace_pomcp.py --games --configs C1-0.1 C2-0.1 \
        --worker $w --num-workers 18 & done; wait
    python3 results_analysis/evaluate_trace_pomcp.py --summary
'''

NUM_SCENARIOS = 100
TASK_LEVELS = (0.3, 1.5)
OUT_DIR = "results_analysis/Results/trace_pomcp"
BELIEF_STEPS = 40
PARTICLES = 50
CONFIGS = {}
for _budget in (0.1, 0.5, 2.0):
    CONFIGS[f"C0-{_budget}"] = {"belief": None, "reuse_tree": True, "time_budget": _budget}
    CONFIGS[f"C1-{_budget}"] = {"belief": "uniform", "reuse_tree": False, "time_budget": _budget}
    CONFIGS[f"C2-{_budget}"] = {"belief": "traces", "reuse_tree": False, "time_budget": _budget}
GAME_FIELDS = ["scenario", "reward", "steps", "completed", "timeout", "coop", "decisions", "sec_per_decision",
               "nsimulations_per_decision", "nrollouts_per_decision"]
BELIEF_FIELDS = ["scenario", "step", "kind", "object", "has_trace", "c1_distance", "c2_distance", "c1_hit", "c2_hit"]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------------------------------------------------------ 1. belief accuracy

def belief_errors(game, agent, belief, seed):
    """
    Draw PARTICLES guesses with the given belief. Returns {(kind, object): (mean distance, hit rate)} for every hidden
    teammate (distance from its guessed to its true cell) and every hidden task (distance from its true cell to the
    nearest guessed hidden task).
    """
    world = TracePlanningEnv.from_real(game, belief=belief)
    world.simulation = True
    random.seed(seed)
    particles = [world.sample_state(agent) for _ in range(PARTICLES)]
    seen_agents = {ag.index for ag in world.components['agents']}
    known_tasks = {task.index for task in world.components['tasks']}
    errors = {}
    for mate in game.env.components['agents']:
        if mate.index == agent.index or mate.index in seen_agents:
            continue
        guesses = [p.get_agent_by_index(mate.index).position for p in particles]
        errors[('teammate', mate.index)] = (np.mean([manhattan(g, mate.position) for g in guesses]),
                                            np.mean([g == mate.position for g in guesses]))
    for task in game.env.components['tasks']:
        if task.completed or task.index in known_tasks:
            continue
        nearest = []
        for p in particles:
            hidden = [t.position for t in p.components['tasks'] if t.index.startswith('H')]
            nearest.append(min((manhattan(h, task.position) for h in hidden), default=2 * DIM))
        errors[('task', task.index)] = (np.mean(nearest), np.mean([d == 0 for d in nearest]))
    return errors, world


def run_belief_accuracy(scenarios):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "belief_accuracy.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BELIEF_FIELDS)
        writer.writeheader()
        for index, scenario in enumerate(scenarios):
            random.seed(index)
            np.random.seed(index)
            game = StigmergicLevelForagingEnv(scenario.copy(), dim=DIM, trace_mode="intent")
            game.reset()
            agent = game.env.get_adhoc_agent()
            for step in range(BELIEF_STEPS):
                state = (random.getstate(), np.random.get_state())  # measuring must not change the game
                c1, _ = belief_errors(game, agent, "uniform", seed=1000 * index + step)
                c2, world = belief_errors(game, agent, "traces", seed=1000 * index + step)
                random.setstate(state[0])
                np.random.set_state(state[1])
                for key in c1:
                    kind, obj = key
                    has_trace = obj in world.reading['teammates'] if kind == 'teammate' else bool(
                        world.reading['bearings'] or world.reading['help'])
                    writer.writerow({"scenario": index, "step": step, "kind": kind, "object": obj,
                                     "has_trace": int(has_trace), "c1_distance": f"{c1[key][0]:.4f}",
                                     "c2_distance": f"{c2[key][0]:.4f}", "c1_hit": f"{c1[key][1]:.4f}",
                                     "c2_hit": f"{c2[key][1]:.4f}"})
                _, _, done, _ = game.step(coop_expert_action(game.env))
                if done:
                    break
            print(f"belief accuracy: scenario {index} done", flush=True)
    print(f"wrote {path}")


# ------------------------------------------------------------------------------------------------ 2. games

def games_path(name, worker):
    return os.path.join(OUT_DIR, "games", f"{name}.w{worker}.csv")


def read_game_rows(name):
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "games", f"{name}.w*.csv"))):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return {int(row["scenario"]): row for row in rows}


def run_games(scenarios, names, worker, num_workers):
    os.makedirs(os.path.join(OUT_DIR, "games"), exist_ok=True)
    for name in names:
        settings = CONFIGS[name]
        done = read_game_rows(name)
        todo = [i for i in range(worker, len(scenarios), num_workers) if i not in done]
        path = games_path(name, worker)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GAME_FIELDS)
            if new_file:
                writer.writeheader()
            for index in todo:
                random.seed(index)
                np.random.seed(index)
                start = time.time()
                result = run_pomcp_episode(scenarios[index], settings["time_budget"], belief=settings["belief"],
                                           reuse_tree=settings["reuse_tree"])
                writer.writerow({"scenario": index, **{key: result[key] for key in GAME_FIELDS if key != "scenario"}})
                f.flush()
                print(f"[{name} w{worker}] scenario {index}: completed={result['completed']} steps={result['steps']} "
                      f"({time.time() - start:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------------------ 3. summary

def wilson(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(np.mean(values)) if len(values) else float("nan"), float("nan")
    return float(np.mean(values)), 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def clustered_ci(rows, values):
    """Mean of values over all rows, and a 95% CI half-width from the spread of the per-scenario means."""
    by_scenario = {}
    for row, value in zip(rows, values):
        by_scenario.setdefault(row["scenario"], []).append(value)
    _, half = mean_ci([np.mean(v) for v in by_scenario.values()])
    return float(np.mean(values)), half


def summarize_beliefs():
    path = os.path.join(OUT_DIR, "belief_accuracy.csv")
    if not os.path.exists(path):
        print("no belief accuracy results yet")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"\n=== Belief accuracy ({path}; {PARTICLES} guesses per decision, first {BELIEF_STEPS} steps, "
          f"teacher-driven) ===")
    print("Distance = Manhattan cells from the truth (teammate: guessed to true cell; task: true cell to nearest guessed "
          "hidden task). Hit = share of guesses exactly right.\nPaired C2 - C1: mean over decisions, 95% CI clustered by "
          "scenario (decisions within a game are correlated, so the CI is computed over per-scenario means).")
    for kind in ("teammate", "task"):
        for label, subset in (("all", None), ("trace sensed", 1), ("no trace sensed", 0)):
            sel = [r for r in rows if r["kind"] == kind and (subset is None or int(r["has_trace"]) == subset)]
            if not sel:
                continue
            c1d = [float(r["c1_distance"]) for r in sel]
            c2d = [float(r["c2_distance"]) for r in sel]
            c1h = [float(r["c1_hit"]) for r in sel]
            c2h = [float(r["c2_hit"]) for r in sel]
            diff, half = clustered_ci(sel, np.subtract(c2d, c1d))
            dh, hh = clustered_ci(sel, np.subtract(c2h, c1h))
            scen = len({r["scenario"] for r in sel})
            print(f"  hidden {kind:8s} | {label:15s} | n={len(sel):5d} ({scen} scenarios) | distance C1 {np.mean(c1d):5.2f}"
                  f" C2 {np.mean(c2d):5.2f} (C2-C1 {diff:+.2f} ± {half:.2f}) | hit C1 {np.mean(c1h):.3f} "
                  f"C2 {np.mean(c2h):.3f} (C2-C1 {dh:+.3f} ± {hh:.3f})")


def summarize_games():
    print("\n=== Games (all tasks completed within 200 steps) ===")
    print(f"{'config':8s} {'n':>4s} {'completed':>10s} {'95% CI':>14s} {'steps':>7s} {'steps|done':>11s} "
          f"{'timeouts':>9s} {'sim/dec':>8s} {'roll/dec':>9s} {'s/dec':>6s}")
    results = {}
    for name in CONFIGS:
        rows = read_game_rows(name)
        if not rows:
            continue
        results[name] = rows
        n = len(rows)
        completed = sum(row["completed"] == "True" for row in rows.values())
        low, high = wilson(completed, n)
        steps = [int(row["steps"]) for row in rows.values()]
        done_steps = [int(row["steps"]) for row in rows.values() if row["completed"] == "True"]
        print(f"{name:8s} {n:4d} {100 * completed / n:9.0f}% [{100 * low:4.0f}, {100 * high:4.0f}]% "
              f"{np.mean(steps):7.1f} {np.mean(done_steps) if done_steps else float('nan'):11.1f} "
              f"{sum(row['timeout'] == 'True' for row in rows.values()):9d} "
              f"{np.mean([float(r['nsimulations_per_decision']) for r in rows.values()]):8.0f} "
              f"{np.mean([float(r['nrollouts_per_decision']) for r in rows.values()]):9.0f} "
              f"{np.mean([float(r['sec_per_decision']) for r in rows.values()]):6.2f}")
    for budget in (0.1, 0.5, 2.0):
        a, b = results.get(f"C1-{budget}"), results.get(f"C2-{budget}")
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        both = [i for i in common if a[i]["completed"] == "True" and b[i]["completed"] == "True"]
        diffs = [int(b[i]["steps"]) - int(a[i]["steps"]) for i in both]
        all_diffs = [int(b[i]["steps"]) - int(a[i]["steps"]) for i in common]
        d, half = mean_ci(diffs)
        da, halfa = mean_ci(all_diffs)
        only_c2 = sum(a[i]["completed"] != "True" and b[i]["completed"] == "True" for i in common)
        only_c1 = sum(a[i]["completed"] == "True" and b[i]["completed"] != "True" for i in common)
        faster = sum(x < 0 for x in diffs)
        slower = sum(x > 0 for x in diffs)
        print(f"\nPaired C2 vs C1 at {budget} s ({len(common)} common scenarios):")
        print(f"  steps C2 - C1, all scenarios (timeouts count 200): {da:+.1f} ± {halfa:.1f}")
        print(f"  steps C2 - C1 when both finish (n={len(both)}): {d:+.1f} ± {half:.1f}; C2 faster in {faster}, "
              f"slower in {slower}, tied in {len(both) - faster - slower}")
        print(f"  only C2 finishes: {only_c2}; only C1 finishes: {only_c1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief", action="store_true", help="measure belief accuracy (C1 vs C2)")
    parser.add_argument("--games", action="store_true", help="play games with POMCP configurations")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=NUM_SCENARIOS)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.belief or args.games:
        scenario_set = create_evaluation_set(args.scenarios, task_level_range=TASK_LEVELS)
        if args.belief:
            run_belief_accuracy(scenario_set)
        if args.games:
            run_games(scenario_set, args.configs, args.worker, args.num_workers)
    if args.summary:
        summarize_beliefs()
        summarize_games()
