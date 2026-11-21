"""
The world POMCP imagines in, with traces (docs/trace_pomcp/trace_pomcp_method.tex, Section 4).

POMCP (src/reasoning/pomcp.py, framework) only talks to its world through four methods:
    copy()            clone the world before imagining a step
    step(action)      imagine one step
    get_observation() / observation_is_equal(obs)   what the ad hoc agent would see
    sample_state(agent)                              draw one guess of the hidden world (a particle)
TracePlanningEnv is a LevelForagingEnv that overrides copy / step / sample_state, so that
    - every imagined world carries a trace field, and imagined steps produce traces with the same rules as the real game
      (emit_intent_traces, shared with StigmergicLevelForagingEnv);
    - the guesses of the hidden world can read the traces the agent senses (belief="traces").
The framework's code is not modified.

Belief modes (which guesses sample_state draws; Section 5):
    "framework": the framework's LevelForagingEnv.sample_state (C0's sampler; traces are carried but not read)
    "uniform":   C1, the corrected belief without traces: hidden teammates are put back on random cells out of view,
                 hidden tasks fill exactly the remaining slots on random never-seen cells allowed by the scenario
                 generator, with levels from the task prior
"""
import random

import numpy as np

from src.communication.traces import TraceField, TRACE_FIELDS
from src.envs.LevelForagingEnv import Agent, LevelForagingEnv, Task
from src.envs.StigmergicLevelForagingEnv import emit_intent_traces
from src.utils.find_visible import find_visible_grids

BELIEFS = ("framework", "uniform", "traces")
AGE_STEP = 0.2  # age added per step by TraceField.decay
TASK_LEVEL_RANGE = (0.3, 1.5)  # the hard task set (prior for the level of a task never seen)
AHEAD_WEIGHT = 2.0  # Rule 1: a candidate cell straight ahead along the bearing weighs 1 + 2 = 3, a side/behind cell 1
CONE_POWER = 4  # Rule 2: weight max(cos, 0) ** 4 (30 degrees off the bearing: 0.56; 60 degrees: 0.06)
KNOWN_TARGET_COS = 0.95  # Rule 2: the bearing points at a known task if the cosine to it exceeds this


class TracePlanningEnv(LevelForagingEnv):
    """
    Extra attributes (set by from_real, shared by reference between copies unless noted):
        traces:      TraceField of this imagined world (own copy per world; evolves in imagined steps)
        sensed:      the trace fields the agent sensed at decision time, zero outside its view (read-only)
        visible:     set of cells the agent saw at decision time (read-only)
        roster:      the teammates' known parameters: list of (index, level, radius, angle) (read-only)
        observation: the observed world at decision time, the starting point of every guess (never mutated)
        belief:      belief mode, one of BELIEFS
    """

    # number of imagined steps and traces emitted in them since the last reset_counters() (for the breakpoint demo/test)
    imagined_steps = 0
    imagined_emissions = 0
    # optional callback observer(world_after, action), called after every imagined step: a programmable
    # breakpoint (results_analysis/demo_pomcp_traces.py uses it to print one imagined step)
    observer = None

    @classmethod
    def reset_counters(cls):
        cls.imagined_steps = 0
        cls.imagined_emissions = 0

    @classmethod
    def from_real(cls, stig_env, belief="framework", task_level_range=TASK_LEVEL_RANGE):
        """
        Build POMCP's world from the real game (a StigmergicLevelForagingEnv in intent mode) at decision time:
        what the ad hoc agent sees (the framework's observable env) plus the traces in its view.
        """
        if belief not in BELIEFS:
            raise ValueError(f"unknown belief '{belief}', expected one of {BELIEFS}")
        if stig_env.trace_mode != "intent":
            raise ValueError("TracePlanningEnv reproduces intent traces only")
        real = stig_env.env
        observed = real.get_observable_env()  # framework: invisible teammates and unseen tasks removed
        env = _as_planning_env(observed)

        visible = set(find_visible_grids(real))
        mask = np.zeros((stig_env.dim, stig_env.dim), dtype=np.float32)
        for (x, y) in visible:
            mask[x, y] = 1.0
        sensed = stig_env.traces.fields * mask  # local sensing: only traces in visible cells

        env.traces = TraceField(dim=stig_env.dim, decay_rate=stig_env.traces.decay_rate,
                                num_fields=len(TRACE_FIELDS["intent"]))
        env.traces.fields = sensed.copy()
        env.traces.emitter = infer_emitted(sensed)
        env.sensed = sensed
        env.visible = visible
        adhoc_index = real.components['adhoc_agent_index']
        env.roster = [(ag.index, ag.level, ag.radius, ag.angle)
                      for ag in real.components['agents'] if ag.index != adhoc_index]
        env.belief = belief
        env.task_level_range = tuple(task_level_range)
        env.reading = read_traces(sensed, env.roster) if belief == "traces" else None
        env.observation = env.copy()
        env.observation.observation = env.observation  # so guesses (copies of the snapshot) can draw guesses too
        return env

    # ------------------------------------------------------------------ the four methods POMCP uses

    def copy(self):
        clone = _as_planning_env(LevelForagingEnv.copy(self))
        for name in ("sensed", "visible", "roster", "belief", "task_level_range", "observation", "reading"):
            setattr(clone, name, getattr(self, name, None))
        clone.traces = self.traces.copy()
        return clone

    def step(self, action):
        """Imagined step: the framework moves the world, then the traces update as in the real game."""
        next_env, reward, done, info = LevelForagingEnv.step(self, action)  # next_env is a copy made inside
        self.traces.decay()
        emitted = emit_intent_traces(self, self.traces)
        self.traces.diffuse()
        next_env.traces = self.traces.copy()
        TracePlanningEnv.imagined_steps += 1
        TracePlanningEnv.imagined_emissions += emitted
        if TracePlanningEnv.observer is not None:
            TracePlanningEnv.observer(next_env, action)
        return next_env, reward, done, info

    def sample_state(self, agent):
        """Draw one guess of the whole world (a particle) from the belief."""
        if self.belief == "framework":
            return LevelForagingEnv.sample_state(self, agent)
        return self._sample_corrected(agent)

    # ------------------------------------------------------------------ the corrected beliefs (C1; C2 adds trace reading)

    def _sample_corrected(self, agent):
        """
        Start from the observed world at decision time (never from another guess), then
          1. put every teammate that is out of view back on a free cell the agent does not see now;
          2. add exactly (remaining tasks - known tasks) hidden tasks on never-seen cells that the scenario generator
             allows (not on the border, not next to another task), with levels from the task prior.
        """
        world = self.observation.copy()
        world.simulation = self.simulation  # a particle is an imagined world (the snapshot itself is not one)
        world.type_knowledge = True  # as the framework's sampler
        world.parameter_knowledge = True
        cells = [(x, y) for x in range(self.shape[0]) for y in range(self.shape[1])]
        occupied = {ag.position for ag in world.components['agents']}
        occupied |= {task.position for task in world.components['tasks'] if not task.completed}
        occupied |= {tuple(obstacle) for obstacle in world.components['obstacles']}

        # 1. hidden teammates: known parameters (roster), unknown position and type
        seen = {ag.index for ag in world.components['agents']}
        for index, level, radius, angle in self.roster:
            if index in seen:
                continue
            candidates = [c for c in cells if c not in self.visible and c not in occupied]
            if not candidates:
                continue
            position = self._place_hidden_teammate(index, candidates)
            world.components['agents'].append(
                Agent(index=index, atype=None, position=position, direction=random.choice(LevelForagingEnv.directions),
                      radius=radius, angle=angle, level=level))
            occupied.add(position)

        # 2. hidden tasks: exactly the missing number, on cells the generator could have used
        known_tasks = [task for task in world.components['tasks'] if not task.completed]
        remaining = agent.smart_parameters.get('ntasks') or 0
        slots = max(0, remaining - len(known_tasks))
        unseen = set(self.get_unknown_positions(agent))  # never seen by the agent (its memory)
        blocked = set(occupied)
        for task in known_tasks:
            blocked |= set(neighbours(task.position))
        candidates = [c for c in cells if c in unseen and c not in blocked and not on_border(c, self.shape)]
        placed = []

        def place(position, level):
            world.components['tasks'].append(Task(index=f'H{len(placed)}', position=position, level=level))
            placed.append(position)
            taken = {position, *neighbours(position)}
            candidates[:] = [c for c in candidates if c not in taken]

        if self.belief == "traces":
            # Rule 3 first (strongest evidence: a task next to a help trace, with level L + d)
            for cell, level in self.reading['help']:
                if len(placed) >= slots:
                    break
                if any(task.position in neighbours(cell) for task in known_tasks):
                    continue  # the failed load was on a task the agent knows
                options = [n for n in neighbours(cell) if n in candidates]
                if options:
                    place(random.choice(options), level)
            # Rule 2: a task along each teammate's bearing, unless it points at a known (or already guessed) task
            for cell, bearing in self.reading['bearings']:
                if len(placed) >= slots:
                    break
                targets = [task.position for task in known_tasks] + placed
                if any(cosine(cell, target, bearing) > KNOWN_TARGET_COS for target in targets):
                    continue
                weights = [max(cosine(cell, c, bearing), 0.0) ** CONE_POWER for c in candidates]
                if sum(weights) > 0:
                    place(random.choices(candidates, weights=weights, k=1)[0], random.uniform(*self.task_level_range))

        while len(placed) < slots and candidates:  # leftover slots: uniform, as in C1
            place(random.choice(candidates), random.uniform(*self.task_level_range))
        return world

    def _place_hidden_teammate(self, index, candidates):
        """
        candidates: free cells out of view.
        C1: uniform. C2 (Rule 1): only cells within Manhattan radius r of the teammate's youngest sensed trace, weighted
        1 + 2 max(cos, 0) toward its bearing; uniform over all candidates if it left no sensed trace or no cell fits.
        """
        if self.belief == "traces" and index in self.reading['teammates']:
            last = self.reading['teammates'][index]
            cell, radius, bearing = last['cell'], last['radius'], last['bearing']
            near = [c for c in candidates if abs(c[0] - cell[0]) + abs(c[1] - cell[1]) <= radius]
            if near:
                weights = [1.0 + AHEAD_WEIGHT * max(cosine(cell, c, bearing), 0.0) for c in near]
                return random.choices(near, weights=weights, k=1)[0]
        return random.choice(candidates)


def _as_planning_env(env):
    """Turn a LevelForagingEnv (fresh from the framework's copy) into a TracePlanningEnv without copying it again."""
    env.__class__ = TracePlanningEnv
    return env


def read_traces(sensed, roster, tol=1e-4):
    """
    Read the sensed traces with their true meaning (C2's decoder; Section 5.5). Only what the agent senses is used:
    intensity, age, level, help and bearing in visible cells. A teammate is recognised by its level (known).
    The ad hoc agent's own traces (its level is in no roster entry) are ignored.
    Returns
        teammates: {index: {cell, k, radius, bearing}} for the youngest trace of each teammate (ties: higher intensity)
            k = steps since it was left (age / 0.2); radius = k, or k + 1 if the trace is a diffused copy (Eq. 7)
        bearings: [(cell, bearing)] the youngest trace with a nonzero bearing of each teammate (Rule 2)
        help: [(cell, task level)] emitted help traces of teammates: a task next to cell with level = L + d (Rule 3)
    """
    intensity, age, level, help_, bearing_x, bearing_y = sensed
    emitted = infer_emitted(sensed)
    reading = {'teammates': {}, 'bearings': [], 'help': []}
    for index, mate_level, _, _ in roster:
        mine = (intensity > 0) & np.isclose(level, mate_level, atol=tol)
        if not mine.any():
            continue
        cells = [tuple(int(v) for v in c) for c in np.argwhere(mine)]
        youngest = min(cells, key=lambda c: (round(float(age[c]), 3), -float(intensity[c])))
        k = int(np.rint(age[youngest] / AGE_STEP))
        reading['teammates'][index] = {
            'cell': youngest, 'k': k, 'radius': k if emitted[youngest] else k + 1,
            'bearing': (float(bearing_x[youngest]), float(bearing_y[youngest]))}
        pointing = [c for c in cells if bearing_x[c] != 0 or bearing_y[c] != 0]
        if pointing:
            cell = min(pointing, key=lambda c: (round(float(age[c]), 3), -float(intensity[c])))
            reading['bearings'].append((cell, (float(bearing_x[cell]), float(bearing_y[cell]))))
        for c in cells:
            if emitted[c] and help_[c] > 0:
                reading['help'].append((c, float(level[c] + help_[c])))
    return reading


def cosine(origin, cell, bearing):
    """cos of the angle between the bearing and the direction origin -> cell (0 if cell == origin or no bearing)."""
    dx, dy = cell[0] - origin[0], cell[1] - origin[1]
    norm = (dx * dx + dy * dy) ** 0.5 * (bearing[0] ** 2 + bearing[1] ** 2) ** 0.5
    return 0.0 if norm == 0 else (dx * bearing[0] + dy * bearing[1]) / norm


def neighbours(cell):
    """The 4 cells next to a cell (may lie outside the grid)."""
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def on_border(cell, shape):
    return cell[0] in (0, shape[0] - 1) or cell[1] in (0, shape[1] - 1)


def infer_emitted(fields, tol=1e-4):
    """
    Which sensed traces were emitted at their cell (not diffused there)? The agent cannot read the emitter mask, but an
    emitted trace left k steps ago has intensity 1 - 0.2k, while a diffused copy always has less (Section 1, Eq. 7).
    """
    intensity, age = fields[0], fields[1]
    k = np.rint(age / AGE_STEP)
    return (intensity > 0) & (intensity >= 1.0 - AGE_STEP * k - tol)
