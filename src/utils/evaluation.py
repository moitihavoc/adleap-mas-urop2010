import contextlib
import io
import random
import time

import numpy as np
import torch

from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.envs.TracePlanningEnv import TracePlanningEnv
from src.reasoning.pomcp import pomcp_planning
from src.utils.scenarios import make_scenario


NUM_SCENARIOS = 100
DIM = 10
NAGENTS = 2
NTASKS = 5
ADHOC_RADIUS = 0.25
ADHOC_ANGLE = 1.0
ADHOC_LEVEL = 0.9
MAX_STEPS = 200

# smart_parameters entries owned by a planner or an estimator; dropped at the start of every POMCP episode
PLANNER_KEYS = ('pomcp', 'count', 'estimation', 'estimation_method', 'estimation_args')


def create_evaluation_set(num_scenarios=NUM_SCENARIOS, seed=2026, task_level_range=(0.0, 1.0)):
    """Create one shared evaluation set with the training distribution (built by make_scenario, like the training data)."""
    random.seed(seed)
    scenarios = []

    for _ in range(num_scenarios):
        scenarios.append(make_scenario(dim=DIM, nagents=NAGENTS, ntasks=NTASKS, adhoc_level=ADHOC_LEVEL,
                                       adhoc_radius=ADHOC_RADIUS, adhoc_angle=ADHOC_ANGLE,
                                       task_level_range=task_level_range))

    return scenarios


def run_evaluation(model, scenarios, ablate=""):
    """Evaluate a model on shared scenario copies and return per-scenario data."""
    rewards = []
    steps = []
    successes = []

    for base_env in scenarios:
        env = StigmergicLevelForagingEnv(base_env.copy(), dim=DIM)
        obs_tensor = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 200:
            input_tensor = obs_tensor.unsqueeze(0).clone()
            if ablate == "no trace":
                input_tensor[:, 3:8, :, :] = 0.0
            elif ablate == "basic":
                input_tensor[:, 5:8, :, :] = 0.0
            elif ablate == "level":
                input_tensor[:, 6:8, :, :] = 0.0

            with torch.no_grad():
                pi_logits, _ = model(input_tensor)
                action = torch.argmax(pi_logits, dim=1).item()

            obs_tensor, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1

        rewards.append(episode_reward)
        steps.append(step)
        successes.append(episode_reward > 0)

    return {
        'rewards': rewards,
        'steps': steps,
        'successes': successes,
    }


def run_episode(base_env, choose_action, trace_mode="intent"):
    """
    Play one episode of a copy of base_env with `choose_action(env, obs_tensor, adhoc_agent) -> int` as the ad hoc
    agent (env is the StigmergicLevelForagingEnv). Returns reward, steps, completed (every task collected), timeout,
    coop (tasks completed by more than one agent).
    """
    env = StigmergicLevelForagingEnv(base_env.copy(), dim=DIM, trace_mode=trace_mode)
    obs_tensor = env.reset()
    adhoc_agent = env.env.get_adhoc_agent()
    ntasks = len(env.env.components['tasks'])
    done, step, total_reward, coop = False, 0, 0, 0
    while not done and step < MAX_STEPS:
        action = choose_action(env, obs_tensor, adhoc_agent)
        obs_tensor, reward, done, info = env.step(action)
        total_reward += reward
        coop += info.get('coop', 0) if info else 0
        step += 1
    return {'reward': total_reward, 'steps': step, 'completed': total_reward >= ntasks,
            'timeout': step >= MAX_STEPS and not done, 'coop': coop}


def run_pomcp_episode(base_env, time_budget, belief=None, reuse_tree=True, max_depth=20, trace_mode="intent"):
    """
    One episode with the framework's POMCP planner (src/reasoning/pomcp.py) as the ad hoc agent, with its own view.
        belief=None: POMCP plans on the framework's observable world, as on post_urop1010 (C0; traces never reach it)
        belief="framework" | "uniform" | "traces": POMCP plans on a TracePlanningEnv built from the real game at every
            decision (traces carried and imagined; see src/envs/TracePlanningEnv.py for what each belief reads)
        reuse_tree: keep the search tree between decisions (the framework's default). With False every decision starts a
            new tree, so the belief is rebuilt from the current observation (and its traces) at every step.
        time_budget: seconds of search per decision; max_depth: tree levels (20 = about 10 steps ahead)
    The planner is cached in the agent's smart_parameters, and Agent.copy() shares that dict between copies, so the
    episode starts by giving the agent its own dict without planner entries (fixes the October planner-sharing bug).
    Returns run_episode's dict plus decisions, sec_per_decision, nsimulations/nrollouts per decision.
    """
    stats = {'decisions': 0, 'seconds': 0.0, 'agent': None}

    def choose_action(env, obs_tensor, adhoc_agent):
        if stats['agent'] is None:
            adhoc_agent.smart_parameters = {key: value for key, value in adhoc_agent.smart_parameters.items()
                                            if key not in PLANNER_KEYS}
            stats['agent'] = adhoc_agent
        if not reuse_tree:
            adhoc_agent.smart_parameters.pop('pomcp', None)
        if belief is None:
            world = env.env.get_observable_env()
        else:
            world = TracePlanningEnv.from_real(env, belief=belief)
        start = time.time()
        with contextlib.redirect_stdout(io.StringIO()):  # the planner prints its Q-table at every decision
            action, _ = pomcp_planning(world, adhoc_agent, max_depth=max_depth, time_budget=time_budget)
        stats['seconds'] += time.time() - start
        stats['decisions'] += 1
        count = adhoc_agent.smart_parameters.get('count', {})
        stats['nsimulations'] = stats.get('nsimulations', 0) + count.get('nsimulations', 0)
        stats['nrollouts'] = stats.get('nrollouts', 0) + count.get('nrollouts', 0)
        if not reuse_tree:
            return action
        # with tree reuse the planner's counters accumulate over the episode; keep only the last totals
        stats['nsimulations'], stats['nrollouts'] = count.get('nsimulations', 0), count.get('nrollouts', 0)
        return action

    result = run_episode(base_env, choose_action, trace_mode)
    decisions = max(stats['decisions'], 1)
    result.update(decisions=stats['decisions'], sec_per_decision=stats['seconds'] / decisions,
                  nsimulations_per_decision=stats.get('nsimulations', 0) / decisions,
                  nrollouts_per_decision=stats.get('nrollouts', 0) / decisions)
    return result


def summarize(values):
    """Return mean and a 95% normal-approximation confidence-interval margin."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, 1.96 * standard_error
