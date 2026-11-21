import argparse
import contextlib
import io
import os
import random
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.envs.TracePlanningEnv import TracePlanningEnv
from src.reasoning.pomcp import pomcp_planning
from src.utils.evaluation import DIM, PLANNER_KEYS, create_evaluation_set

'''
=== DEMO: does POMCP see and imagine traces? (the instructor's "breakpoint") ===
Explained in docs/trace_pomcp/trace_pomcp_method.tex, Sections 3-4.

1. Find a moment in a real game where the ad hoc agent senses a teammate's trace but cannot see that teammate
   (the ad hoc agent stays idle, pressing load, while the teammates play).
2. Build POMCP's world from that moment (TracePlanningEnv.from_real) and print it next to the real world.
3. Run one POMCP decision and catch the first imagined step with TracePlanningEnv.observer (a programmable breakpoint):
   print the imagined world with its new traces, and count imagined steps / traces.

4. --figure: at the same moment, draw 2,000 guesses with C1 (no traces) and C2 (reads traces) and plot where they put
   the hidden teammate and the hidden tasks, with the truth marked -> results_analysis/Plots/pomcp_trace_belief.png

Usage:
    python3 results_analysis/demo_pomcp_traces.py              # breakpoint demo, evaluation scenario chosen automatically
    python3 results_analysis/demo_pomcp_traces.py --scenario 3
    python3 results_analysis/demo_pomcp_traces.py --figure
'''

TEAMMATE_SYMBOL = {'w1': 'B', 'w2': 'C'}  # letters, so they cannot be confused with trace intensities


def find_trace_moment(scenarios, max_steps=30, start=0):
    """First (scenario, step) where the agent senses a teammate trace while that teammate is out of view."""
    for index in range(start, len(scenarios)):
        random.seed(index)
        np.random.seed(index)
        game = StigmergicLevelForagingEnv(scenarios[index].copy(), dim=DIM, trace_mode="intent")
        game.reset()
        for step in range(max_steps):
            game.step(4)
            world = TracePlanningEnv.from_real(game)
            seen = {ag.index for ag in world.components['agents']}
            for index_, level, _, _ in world.roster:
                sensed_levels = world.sensed[2][world.sensed[0] > 0]
                if index_ not in seen and np.any(np.isclose(sensed_levels, level)):
                    return index, step + 1, game, world, index_
    raise RuntimeError("no such moment found")


def draw(world_components, fields, visible=None, title=""):
    """
    Text picture of a world: agents (A = ad hoc, B = w1, C = w2; 'A*' = standing on a trace it just emitted), tasks (T),
    trace intensity (1-9 = 0.1-0.9, * = 1.0), '?' = cell the agent does not see.
    """
    grid = [['  .' for _ in range(DIM)] for _ in range(DIM)]
    for x in range(DIM):
        for y in range(DIM):
            if visible is not None and (x, y) not in visible:
                grid[x][y] = '  ?'
            if fields[0, x, y] > 0:
                value = fields[0, x, y]
                grid[x][y] = '  *' if value > 0.99 else f'  {int(round(value * 10))}'
    for task in world_components['tasks']:
        if not task.completed:
            grid[task.position[0]][task.position[1]] = '  T'
    for agent in world_components['agents']:
        symbol = 'A' if agent.index == world_components['adhoc_agent_index'] else TEAMMATE_SYMBOL.get(agent.index, 'w')
        x, y = agent.position
        fresh = fields[0, x, y] > 0.99  # the agent emitted a trace at its cell this step
        grid[x][y] = f' {symbol}*' if fresh else f'  {symbol}'
    lines = [title, '   y']
    for y in reversed(range(DIM)):
        lines.append(f'{y:4d}' + ''.join(grid[x][y] for x in range(DIM)))
    lines.append('    ' + ''.join(f'{x:3d}' for x in range(DIM)) + '  x')
    return '\n'.join(lines)


def describe_traces(fields, cells):
    rows = []
    for (x, y) in sorted(cells):
        if fields[0, x, y] > 0:
            rows.append(f'    cell {(x, y)}: intensity {fields[0, x, y]:.2f}, age {fields[1, x, y]:.1f}, '
                        f'level {fields[2, x, y]:.2f}, help {fields[3, x, y]:.2f}, '
                        f'bearing ({fields[4, x, y]:+.2f}, {fields[5, x, y]:+.2f})')
    return '\n'.join(rows) if rows else '    (none)'


def breakpoint_demo(scenario=None, budget=0.1):
    scenarios = create_evaluation_set(100, task_level_range=(0.3, 1.5))
    index, step, game, world, hidden = find_trace_moment(scenarios, start=scenario or 0)
    real = game.env
    print(f'Evaluation scenario {index}, after {step} steps of the ad hoc agent pressing load.')
    print(f'Teammate {hidden} is out of view, but the agent senses one of its traces.\n')
    print(draw(real.components, game.traces.fields, title='REAL WORLD (the truth, all traces):'))
    print()
    print(draw(world.components, world.traces.fields, world.visible,
               title="POMCP'S WORLD at decision time (what the agent sees: own view + traces in view):"))
    print('  sensed traces:')
    print(describe_traces(world.sensed, world.visible))
    real_ids = sorted(ag.index for ag in real.components['agents'])
    pomcp_ids = sorted(ag.index for ag in world.components['agents'])
    print(f'\n  agents in the real world: {real_ids};  in POMCP\'s world: {pomcp_ids}')
    print(f'  tasks in the real world: {sum(not t.completed for t in real.components["tasks"])};  '
          f'in POMCP\'s world: {len(world.components["tasks"])} (the rest are hidden and must be guessed)')

    caught = []

    def observer(world_after, action):
        # catch the first imagined step that leaves a fresh trace (a non-help trace cannot overwrite a cell holding a
        # help trace, so some imagined steps leave none)
        if not caught and (world_after.traces.fields[0] > 0.99).any():
            caught.append((world_after.copy(), action))

    adhoc = real.get_adhoc_agent()
    adhoc.smart_parameters = {k: v for k, v in adhoc.smart_parameters.items() if k not in PLANNER_KEYS}
    TracePlanningEnv.reset_counters()
    TracePlanningEnv.observer = observer
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            action, _ = pomcp_planning(world, adhoc, max_depth=20, time_budget=budget)
    finally:
        TracePlanningEnv.observer = None

    imagined, imagined_action = caught[0]
    names = {0: 'East', 1: 'West', 2: 'North', 3: 'South', 4: 'Load'}
    print('\n--- BREAKPOINT: the first imagined step that left a fresh trace ---')
    print(draw(imagined.components, imagined.traces.fields,
               title=f"IMAGINED WORLD after the ad hoc agent imagines '{names[imagined_action]}' "
                     f"(* = trace emitted in the imagined step):"))
    fresh = [(x, y) for x in range(DIM) for y in range(DIM) if imagined.traces.fields[0, x, y] > 0.99]
    print('  fresh imagined traces:')
    print(describe_traces(imagined.traces.fields, fresh))
    print(f'\nOne decision ({budget} s of search): POMCP imagined {TracePlanningEnv.imagined_steps} steps '
          f'and {TracePlanningEnv.imagined_emissions} traces; it chose {names[action]}.')
    print('Note: the hidden teammate is missing from POMCP\'s world, so it never imagines that teammate\'s traces; '
          'the sensed trace is carried but not read (belief "framework").')


SEQUENTIAL = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # blue ramp
INK, MUTED_INK, GRID_LINE = '#0b0b0b', '#52514e', '#e6e5e0'
FIGURE_PATH = 'results_analysis/Plots/pomcp_trace_belief.png'


def belief_maps(game, belief, n, seed=0):
    """Share of n guesses putting each hidden teammate on each cell, and mean number of guessed hidden tasks per cell."""
    world = TracePlanningEnv.from_real(game, belief=belief)
    world.simulation = True
    agent = game.env.get_adhoc_agent()
    seen = {ag.index for ag in world.components['agents']}
    mates = {index: np.zeros((DIM, DIM)) for index, _, _, _ in world.roster if index not in seen}
    tasks = np.zeros((DIM, DIM))
    random.seed(seed)
    for _ in range(n):
        particle = world.sample_state(agent)
        for index in mates:
            x, y = particle.get_agent_by_index(index).position
            mates[index][x, y] += 1 / n
        for task in particle.components['tasks']:
            if task.index.startswith('H'):
                tasks[task.position] += 1 / n
    return world, mates, tasks


def belief_figure(scenario=None, n=2000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    scenarios = create_evaluation_set(100, task_level_range=(0.3, 1.5))
    index, step, game, _, hidden = find_trace_moment(scenarios, start=scenario or 0)
    real = game.env
    cmap = LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    maps = {belief: belief_maps(game, belief, n) for belief in ('uniform', 'traces')}
    world = maps['traces'][0]
    known = {task.index for task in world.components['tasks']}
    hidden_tasks = [t for t in real.components['tasks'] if not t.completed and t.index not in known]
    mate_truth = real.get_agent_by_index(hidden).position
    adhoc = real.get_adhoc_agent().position
    sensed_cells = [(x, y) for x in range(DIM) for y in range(DIM)
                    if world.sensed[0, x, y] > 0 and abs(world.sensed[2, x, y] - real.get_agent_by_index(hidden).level) < 1e-4]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0), facecolor='#fcfcfb', constrained_layout=True)
    rows = [('mate', f'Hidden teammate {hidden}: share of guesses per cell', 'share of guesses'),
            ('task', 'Hidden tasks: guessed tasks per cell (mean over guesses)', 'guessed tasks per cell')]
    columns = [('uniform', 'C1: belief ignores traces'), ('traces', 'C2: belief reads traces')]
    for r, (kind, row_title, bar_label) in enumerate(rows):
        values = {b: (maps[b][1][hidden] if kind == 'mate' else maps[b][2]) for b, _ in columns}
        vmax = max(v.max() for v in values.values())
        for c, (belief, col_title) in enumerate(columns):
            ax = axes[r, c]
            image = ax.imshow(values[belief].T, origin='lower', cmap=cmap, vmin=0, vmax=vmax,
                              extent=(-0.5, DIM - 0.5, -0.5, DIM - 0.5))
            for (x, y) in world.visible:  # cells in the agent's view: hatched, nothing hidden can be there now
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, hatch='////', edgecolor=GRID_LINE,
                                           linewidth=0))
            ax.text(*adhoc, 'A', ha='center', va='center', fontsize=11, fontweight='bold', color=INK)
            if kind == 'mate':
                for (x, y) in sensed_cells:
                    ax.plot(x, y, marker='o', markersize=4, color=MUTED_INK, linestyle='none')
                ax.plot(*mate_truth, marker='o', markersize=17, markerfacecolor='none', markeredgecolor=INK,
                        markeredgewidth=2)
                hit = values[belief][mate_truth]
                note = f'{100 * hit:.0f}% of guesses on the true cell'
            else:
                for task in hidden_tasks:
                    ax.plot(*task.position, marker='s', markersize=15, markerfacecolor='none', markeredgecolor=INK,
                            markeredgewidth=2)
                for task in world.components['tasks']:
                    ax.text(*task.position, 'T', ha='center', va='center', fontsize=10, color=INK)
                target = tuple(real.get_agent_by_index(hidden).target)
                others = np.mean([values[belief][t.position] for t in hidden_tasks if t.position != target])
                note = f'{hidden}\'s target {target}: {values[belief][target]:.2f}; other hidden tasks: {others:.2f}'
            ax.set_title(f'{col_title}\n{note}', fontsize=10, color=INK, loc='left')
            ax.set_xticks(range(DIM))
            ax.set_yticks(range(DIM))
            ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
            for spine in ax.spines.values():
                spine.set_color(GRID_LINE)
        bar = fig.colorbar(image, ax=axes[r, :].tolist(), shrink=0.85, pad=0.02)
        bar.set_label(bar_label, color=MUTED_INK, fontsize=9)
        bar.ax.tick_params(colors=MUTED_INK, labelsize=8)
        bar.outline.set_visible(False)
        axes[r, 0].set_ylabel(row_title, color=INK, fontsize=10)
    fig.suptitle(f'Where POMCP guesses the hidden world: evaluation scenario {index}, step {step} '
                 f'({n:,} guesses per belief)', color=INK, fontsize=12, x=0.02, ha='left')
    fig.text(0.02, -0.035, 'A = ad hoc agent; hatched = cells in its view; T = tasks it sees; small dots = sensed traces '
             f'of {hidden}; circle = true position of {hidden}; squares = true hidden tasks (never seen).',
             color=MUTED_INK, fontsize=8.5, ha='left')
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches='tight', facecolor='#fcfcfb')
    print(f'wrote {FIGURE_PATH}')
    target = tuple(real.get_agent_by_index(hidden).target)
    for belief in ('uniform', 'traces'):
        print(f'  {belief}: {100 * maps[belief][1][hidden][mate_truth]:.1f}% of guesses put {hidden} on its true cell '
              f'{mate_truth}; guessed tasks on its target {target}: {maps[belief][2][target]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=None, help='first evaluation scenario to search from')
    parser.add_argument('--budget', type=float, default=0.1, help='seconds of search for the demo decision')
    parser.add_argument('--figure', action='store_true', help='draw the C1 vs C2 belief figure instead')
    args = parser.parse_args()
    if args.figure:
        belief_figure(args.scenario)
    else:
        breakpoint_demo(args.scenario, args.budget)
