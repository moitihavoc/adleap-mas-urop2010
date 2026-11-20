import numpy as np

from src.reasoning.a_star import a_star_planning
from src.reasoning.levelbased.l1 import build_state

"""
Privileged cooperative teacher for the ad hoc agent (used to label data for behavior cloning / DAgger).

It reads the real environment, i.e. information the learned student never observes directly:
the true task levels and the teammates' levels and current targets. Each step it re-chooses a task:
    - skip a task the committed teammates can already finish on their own (no duplicated effort);
    - join a task some teammates are committed to but cannot finish, when the ad hoc agent's level is enough;
    - otherwise go solo on a task the ad hoc agent can finish alone;
    - if nothing is feasible, go to the nearest open task and wait there for help.
Movement and loading follow l1_planning: A* towards the task, press load (4) once facing it.
"""

SOLO_PENALTY = 3  # extra cost (in cells) of a solo task: joining a teammate wins unless it is more than 3 cells farther


def facing_cell(agent):
    x, y = agent.position
    if agent.direction == np.pi / 2:
        return (x, y + 1)
    if agent.direction == 3 * np.pi / 2:
        return (x, y - 1)
    if agent.direction == 0:
        return (x + 1, y)
    if agent.direction == np.pi:
        return (x - 1, y)
    return None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def committed_level(task, teammates):
    """Sum of the levels of the teammates whose current target is this task."""
    return sum(teammate.level for teammate in teammates
               if teammate.target is not None and teammate.target in (task.position, task.index))


def choose_target(env):
    """Return the task the ad hoc agent should work on, or None when every task is completed."""
    adhoc = env.get_adhoc_agent()
    teammates = [agent for agent in env.components['agents'] if agent.index != adhoc.index]

    best, best_cost = None, None
    nearest, nearest_distance = None, None
    for task in env.components['tasks']:
        if task.completed:
            continue
        distance = manhattan(adhoc.position, task.position)
        if nearest is None or distance < nearest_distance:
            nearest, nearest_distance = task, distance

        committed = committed_level(task, teammates)
        if committed >= task.level:
            continue                                    # teammates finish it alone
        if committed > 0 and committed + adhoc.level >= task.level:
            cost = distance                             # join
        elif adhoc.level >= task.level:
            cost = distance + SOLO_PENALTY              # solo
        else:
            continue                                    # the ad hoc agent cannot make it happen now

        if best is None or cost < best_cost:
            best, best_cost = task, cost

    return best if best is not None else nearest


def coop_expert_action(env):
    """Action (0-4) of the cooperative teacher in the real (fully known) LevelForagingEnv."""
    adhoc = env.get_adhoc_agent()
    task = choose_target(env)
    if task is None:
        return 4
    if facing_cell(adhoc) == task.position:
        return 4
    state = build_state(env.shape, env.components)
    return a_star_planning(state, env.shape[0], env.shape[1], env.action_space,
                           adhoc.position, task.position)
