import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task, there_is_task
from src.communication.traces import TraceField, TRACE_FIELDS
from src.utils.tensor_convert import NUM_SELF_FEATURES, add_self_features, dict_to_tensor
from src.utils.find_visible import find_visible_grids

# --- Wrapper for the LevelForagingEnv to include stigmergic communication

"""
What this wapper does:
- Redefines the observation space to include the trace field information.
- Modify the reset and step methods to update the trace field based on agent actions and task states.

Trace modes (every agent leaves a mark at its own cell after every action in both modes):
- "original": help/claim bits compare the emitter's level with the first task it sees (the setup of the report).
- "intent":   help is the level deficit measured on a failed load attempt (task level - own level),
              and the payload is the bearing from the emitter to its current target.
              The ad hoc agent's own bearing is always (0, 0): a learned policy has no target to report.
local_traces: the ad hoc agent only senses traces inside its field of view (default: on for "intent", off for "original").
self_features: levels of visible teammates and tasks, plus own-position and heading planes (src/utils/tensor_convert.py).
"""

class StigmergicLevelForagingEnv(gym.Wrapper):
    def __init__(self, env: LevelForagingEnv, dim:int, decay_rate: float = 0.2,
                 trace_mode: str = "original", local_traces: bool = None, self_features: bool = False):
        super().__init__(env)
        if trace_mode not in TRACE_FIELDS:
            raise ValueError(f"unknown trace_mode '{trace_mode}', expected one of {list(TRACE_FIELDS)}")
        self.dim = dim
        self.decay = decay_rate
        self.trace_mode = trace_mode
        self.local_traces = (trace_mode == "intent") if local_traces is None else local_traces
        self.self_features = self_features  # levels of visible teammates/tasks + own position and heading planes
        num_fields = len(TRACE_FIELDS[trace_mode])
        self.traces = TraceField(dim=dim, decay_rate=decay_rate, num_fields=num_fields)
        # redefine the observation space to include the trace field information
        self.observation_space = spaces.Box(
            low = -1.0 if trace_mode == "intent" else 0.0, high = 1.0,  # bearing components lie in [-1, 1]
            shape = (3 + num_fields + 1 + (NUM_SELF_FEATURES if self_features else 0), dim, dim), # 3 for agents/tasks/obstacles, the trace fields, 1 for visibility, self features
            dtype = np.float32
        )

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs).get_observation()
        self.traces.reset()
        visible_grids = find_visible_grids(self.env)
        obs_tensor = dict_to_tensor(obs_dict, self.traces.fields, self.dim, visible_grids, self.local_traces)
        if self.self_features:
            obs_tensor = add_self_features(obs_tensor, obs_dict, self.env)
        return obs_tensor

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.traces.decay()

        # construct a tensor based on observable components
        visible_grids = find_visible_grids(self.env)

        # emit traces for every agent in the environment
        if self.trace_mode == "intent":
            emit_intent_traces(self.env, self.traces)
        else:
            self._emit_original_traces()

        self.traces.diffuse()

        obs_dict = state.get_observation()
        obs_tensor = dict_to_tensor(obs_dict, self.traces.fields, self.dim, visible_grids, self.local_traces)
        if self.self_features:
            obs_tensor = add_self_features(obs_tensor, obs_dict, self.env)

        return obs_tensor, reward, done, info

    def _emit_original_traces(self):
        for agent in self.env.components['agents']:
            level = agent.level
            obs_state = self.env.get_visible_components(agent) # get the observable state for each agent
            task_level = -1.0
            help_signal = 0
            claim = 0
            for task in self.env.components['tasks']:
                if obs_state['tasks'] and task.index == obs_state['tasks'][0][0]:
                    task_level = task.level
                    break

            if task_level >= 0:
                help_signal = 1 if level < task_level else 0
                claim = 1 if level >= task_level else 0

            self.traces.fusion(level, help_signal, claim, agent.position)


# --- Trace rules shared by the real game (the wrapper above) and POMCP's imagined copies (src/envs/TracePlanningEnv.py),
# so both produce traces in exactly the same way.

def emit_intent_traces(env, traces):
    """
    Every agent in env leaves an intent trace at its own cell (call after env.step, between decay and diffuse).
        help: the level deficit min(task level - own level, 1) if the agent pressed load facing a task that is still
              there after the step (a failed load attempt), else 0
        bearing: unit vector from the agent to its current target; always (0, 0) for the ad hoc agent
    Returns the number of traces emitted.
    """
    adhoc_index = env.components['adhoc_agent_index']
    emitted = 0
    for agent in env.components['agents']:
        # a load attempt failed if the agent pressed load facing a task that is still there after the step
        help_deficit = 0.0
        if agent.next_action == 4:
            task = there_is_task(env, agent.position, agent.direction)
            if task is not None:
                help_deficit = min(max(task.level - agent.level, 0.0), 1.0)

        bearing = (0.0, 0.0) if agent.index == adhoc_index else bearing_to_target(env, agent)
        traces.fusion(agent.level, help_deficit, bearing, agent.position)
        emitted += 1
    return emitted


def bearing_to_target(env, agent):
    """Unit vector from the agent to its current target (a position or a task index), (0, 0) if none."""
    target = agent.target
    if target is not None and not isinstance(target, (tuple, list)):
        target = next((task.position for task in env.components['tasks']
                       if task.index == target), None)
    if target is None:
        return (0.0, 0.0)
    dx, dy = target[0] - agent.position[0], target[1] - agent.position[1]
    norm = math.hypot(dx, dy)
    return (0.0, 0.0) if norm == 0 else (dx / norm, dy / norm)
