"""
Tests for POMCP's imagined world with traces (src/envs/TracePlanningEnv.py).
Section numbers refer to docs/trace_pomcp/trace_pomcp_method.tex.
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.communication.traces import TraceField
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.envs.TracePlanningEnv import TracePlanningEnv, infer_emitted
from src.reasoning.pomcp import pomcp_planning

DIM = 10


def make_game(steps=0, seed=0):
    """
    A hand-built real game (intent traces, local sensing):
        A  (ad hoc, level 0.9, view radius 0.25 = about 3.5 cells) at (1, 1)
        w1 (level 0.6) at (8, 8): far outside A's view
        w2 (level 0.7) at (2, 3): inside A's view
        T0 (level 1.3) at (4, 1): inside A's view, three cells away (not adjacent)
        T1 (level 0.5) at (7, 5), T2 (level 1.1) at (5, 8): outside A's view
    After `steps` steps of A pressing load (A stays put), the teammates have moved and left traces.
    """
    random.seed(seed)
    np.random.seed(seed)
    components = {
        'agents': [
            Agent(index='A', atype='l1', position=(1, 1), direction=np.pi / 2, radius=0.25, angle=1.0, level=0.9),
            Agent(index='w1', atype='l1', position=(8, 8), direction=np.pi, radius=1.0, angle=1.0, level=0.6),
            Agent(index='w2', atype='l2', position=(2, 3), direction=np.pi, radius=1.0, angle=1.0, level=0.7),
        ],
        'adhoc_agent_index': 'A',
        'template_types': ['l1', 'l2'],
        'tasks': [Task(index='T0', position=(4, 1), level=1.3),
                  Task(index='T1', position=(7, 5), level=0.5),
                  Task(index='T2', position=(5, 8), level=1.1)],
        'impostor_index': None,
        'obstacles': [],
    }
    base_env = LevelForagingEnv(shape=(DIM, DIM), components=components, display=False,
                                type_knowledge=False, parameter_knowledge=True, vision_block=False)
    game = StigmergicLevelForagingEnv(base_env, dim=DIM, trace_mode="intent")
    game.reset()
    for _ in range(steps):
        game.step(4)
    return game


def planning_world(game, belief="framework"):
    world = TracePlanningEnv.from_real(game, belief=belief)
    world.simulation = True  # as pomcp_planning does
    return world


# ---------------------------------------------------------------- Stage 2: the world POMCP imagines in (Section 4)

def test_world_starts_from_what_the_agent_senses():
    game = make_game(steps=3)
    world = planning_world(game)
    visible = world.visible
    # a teammate is in POMCP's world exactly when it stands in a visible cell (the framework drops the others)
    indices = {ag.index for ag in world.components['agents']}
    for index in ('w1', 'w2'):
        assert (index in indices) == (game.env.get_agent_by_index(index).position in visible)
    assert 'w1' not in indices  # starts at (8, 8), far outside the view
    # traces: equal to the real game's inside the view, zero outside (local sensing)
    for x in range(DIM):
        for y in range(DIM):
            if (x, y) in visible:
                assert np.allclose(world.traces.fields[:, x, y], game.traces.fields[:, x, y])
            else:
                assert np.all(world.traces.fields[:, x, y] == 0)
    assert {r[0] for r in world.roster} == {'w1', 'w2'}


def test_visible_task_level_is_known_anywhere_in_the_view():
    """User's requirement: a task inside the view is seen with its true level, not only when adjacent to it."""
    world = planning_world(make_game())
    tasks = {t.index: t for t in world.components['tasks']}
    assert 'T0' in tasks and tasks['T0'].level == pytest.approx(1.3)
    assert tasks['T0'].position == (4, 1)  # three cells from the agent at (1, 1)
    assert 'T1' not in tasks and 'T2' not in tasks  # outside the view: unknown


def test_copy_is_a_planning_world_with_independent_traces():
    world = planning_world(make_game(steps=2))
    clone = world.copy()
    assert isinstance(clone, TracePlanningEnv)
    assert clone.roster is world.roster and clone.observation is world.observation
    clone.traces.fields[:] = 0.5
    clone.traces.emitter[:] = True
    assert not np.allclose(world.traces.fields, 0.5)
    assert not world.traces.emitter.all()


def test_imagined_step_leaves_a_trace_at_every_agent_cell():
    """The instructor's breakpoint, automated: inside POMCP's world an imagined step produces traces."""
    world = planning_world(make_game(steps=2))
    TracePlanningEnv.reset_counters()
    next_world, _, _, _ = world.copy().step(0)
    assert isinstance(next_world, TracePlanningEnv)
    assert TracePlanningEnv.imagined_steps == 1
    assert TracePlanningEnv.imagined_emissions == len(next_world.components['agents'])
    for agent in next_world.components['agents']:
        x, y = agent.position
        assert next_world.traces.fields[0, x, y] == pytest.approx(1.0)  # fresh trace
        assert next_world.traces.fields[2, x, y] == pytest.approx(agent.level)  # emitter level


def test_infer_emitted_tells_emitted_from_diffused_traces():
    field = TraceField(dim=DIM, num_fields=6)
    field.emit(0.6, 0.0, (1.0, 0.0), (5, 5))
    field.diffuse()
    for _ in range(2):  # two steps later, nobody re-emits
        field.decay()
        field.diffuse()
    inferred = infer_emitted(field.fields)
    assert inferred[5, 5]
    assert field.fields[0, 6, 5] > 0 and not inferred[6, 5]  # a diffused copy next to it
    assert np.array_equal(inferred, field.emitter & (field.fields[0] > 0))


def test_one_pomcp_decision_runs_on_the_planning_world():
    game = make_game(steps=1)
    adhoc = game.env.get_adhoc_agent()
    adhoc.smart_parameters = {k: v for k, v in adhoc.smart_parameters.items() if k not in ('pomcp', 'count')}
    world = TracePlanningEnv.from_real(game, belief="framework")
    TracePlanningEnv.reset_counters()
    action, _ = pomcp_planning(world, adhoc, max_depth=10, time_budget=0.05)
    assert action in (0, 1, 2, 3, 4)
    assert TracePlanningEnv.imagined_steps > 0 and TracePlanningEnv.imagined_emissions > 0


# ---------------------------------------------------------------- Stage 3: the corrected belief without traces (C1)

def draw_particles(game, belief, n, seed=1):
    world = planning_world(game, belief)
    agent = game.env.get_adhoc_agent()
    random.seed(seed)
    return world, agent, [world.sample_state(agent) for _ in range(n)]


def test_remaining_task_count_is_known_during_a_game():
    """The belief fills exactly (remaining - known) task slots; the count it relies on must be right."""
    from src.utils.evaluation import create_evaluation_set
    scenario = create_evaluation_set(1, task_level_range=(0.3, 1.5))[0]
    random.seed(0)
    np.random.seed(0)
    game = StigmergicLevelForagingEnv(scenario.copy(), dim=DIM, trace_mode="intent")
    game.reset()
    for _ in range(60):
        game.step(4)
        remaining = sum(not task.completed for task in game.env.components['tasks'])
        assert game.env.get_adhoc_agent().smart_parameters['ntasks'] == remaining


def test_uniform_particle_is_an_imagined_planning_world():
    game = make_game(steps=3)
    world, agent, particles = draw_particles(game, "uniform", 5)
    for particle in particles:
        assert isinstance(particle, TracePlanningEnv)
        assert particle.simulation  # stepping it must not act like the real game
        assert np.array_equal(particle.traces.fields, world.sensed)


def test_uniform_belief_puts_hidden_teammates_back_out_of_view():
    game = make_game(steps=3)
    world, agent, particles = draw_particles(game, "uniform", 50)
    true_positions = {ag.index: ag.position for ag in game.env.components['agents']}
    seen = {ag.index for ag in world.components['agents']}
    for particle in particles:
        agents = {ag.index: ag for ag in particle.components['agents']}
        assert set(agents) == {'A', 'w1', 'w2'}
        positions = [ag.position for ag in agents.values()]
        assert len(set(positions)) == len(positions)  # no two agents on one cell
        for index, level, _, _ in world.roster:
            assert agents[index].level == pytest.approx(level)
            if index in seen:
                assert agents[index].position == true_positions[index]
            else:
                assert agents[index].position not in world.visible
                assert agents[index].type is None  # the teammates' algorithms stay unknown


def test_uniform_belief_fills_exactly_the_missing_tasks_where_the_generator_allows():
    game = make_game(steps=3)
    world, agent, particles = draw_particles(game, "uniform", 50)
    remaining = agent.smart_parameters['ntasks']
    known = {task.index for task in world.components['tasks']}
    unseen = set(world.get_unknown_positions(agent))
    for particle in particles:
        tasks = [task for task in particle.components['tasks'] if not task.completed]
        assert len(tasks) == remaining
        positions = [task.position for task in tasks]
        for task in tasks:
            assert all(nb not in positions for nb in [(task.position[0] + dx, task.position[1] + dy)
                                                      for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))])
            if task.index in known:
                continue
            assert 0.3 <= task.level <= 1.5
            assert task.position in unseen
            assert 0 < task.position[0] < DIM - 1 and 0 < task.position[1] < DIM - 1
            assert task.position not in {ag.position for ag in particle.components['agents']}


def test_a_guess_drawn_from_a_guess_starts_again_from_the_observation():
    game = make_game(steps=3)
    world, agent, particles = draw_particles(game, "uniform", 1)
    again = particles[0].sample_state(agent)
    assert len(again.components['tasks']) == len(particles[0].components['tasks'])
    assert len(again.components['agents']) == 3


def test_uniform_belief_ignores_traces():
    game = make_game(steps=3)
    world = planning_world(game, "uniform")
    blind = world.copy()
    blind.sensed = np.zeros_like(world.sensed)
    agent = game.env.get_adhoc_agent()
    random.seed(7)
    a = [(ag.index, ag.position) for p in [world.sample_state(agent) for _ in range(20)] for ag in p.components['agents']]
    random.seed(7)
    b = [(ag.index, ag.position) for p in [blind.sample_state(agent) for _ in range(20)] for ag in p.components['agents']]
    assert a == b


# ---------------------------------------------------------------- Stage 4: the belief that reads traces (C2)

from collections import Counter

from src.envs.TracePlanningEnv import cosine, read_traces


def sensed_from(field, cells):
    """The fields of a TraceField as the agent would sense them if it saw exactly `cells`."""
    mask = np.zeros((DIM, DIM), dtype=np.float32)
    for (x, y) in cells:
        mask[x, y] = 1.0
    return field.fields * mask


def test_read_traces_youngest_trace_steps_and_radius():
    field = TraceField(dim=DIM, num_fields=6)
    for x in (2, 3, 4):  # teammate (level 0.6) walks East along row 4, heading East, then leaves
        field.decay()
        field.emit(0.6, 0.0, (1.0, 0.0), (x, 4))
        field.diffuse()
    field.decay()  # one more step: it is now out of sight
    field.diffuse()
    every_cell = [(x, y) for x in range(DIM) for y in range(DIM)]
    reading = read_traces(sensed_from(field, every_cell), [('w1', 0.6, 1.0, 1.0)])
    last = reading['teammates']['w1']
    assert last['cell'] == (4, 4) and last['k'] == 1 and last['radius'] == 1
    assert last['bearing'] == pytest.approx((1.0, 0.0))
    # seeing only a diffused copy next to it widens the radius by one (Eq. 7)
    reading = read_traces(sensed_from(field, [(4, 5)]), [('w1', 0.6, 1.0, 1.0)])
    assert reading['teammates']['w1']['cell'] == (4, 5) and reading['teammates']['w1']['radius'] == 2


def test_read_traces_ignores_the_ad_hoc_agents_own_traces():
    field = TraceField(dim=DIM, num_fields=6)
    field.emit(0.9, 0.3, (0.0, 0.0), (2, 2))  # ad hoc agent (level 0.9) failed a load
    reading = read_traces(field.fields, [('w1', 0.6, 1.0, 1.0)])
    assert reading == {'teammates': {}, 'bearings': [], 'help': []}


def test_rule1_worked_example():
    """Section 5.5 worked example: trace at (4,4), age 0.2, bearing East; (4,5) is visible and empty."""
    world = planning_world(make_game(), "traces")
    world.reading = {'teammates': {'w1': {'cell': (4, 4), 'k': 1, 'radius': 1, 'bearing': (1.0, 0.0)}},
                     'bearings': [], 'help': []}
    candidates = [(4, 4), (5, 4), (3, 4), (4, 3), (9, 9), (0, 9), (6, 6)]  # free cells out of view; (4, 5) is visible
    random.seed(3)
    draws = Counter(world._place_hidden_teammate('w1', candidates) for _ in range(6000))
    assert set(draws) == {(4, 4), (5, 4), (3, 4), (4, 3)}  # never farther than r = 1
    assert draws[(5, 4)] / 6000 == pytest.approx(3 / 6, abs=0.03)
    for cell in ((4, 4), (3, 4), (4, 3)):
        assert draws[cell] / 6000 == pytest.approx(1 / 6, abs=0.03)


def test_rule1_falls_back_to_uniform_without_a_trace():
    world = planning_world(make_game(), "traces")
    world.reading = {'teammates': {}, 'bearings': [], 'help': []}
    random.seed(4)
    draws = Counter(world._place_hidden_teammate('w1', [(9, 9), (0, 9)]) for _ in range(2000))
    assert draws[(9, 9)] / 2000 == pytest.approx(0.5, abs=0.05)


def uniform_and_traces_particles(game, reading, n=40, seed=5):
    traced = planning_world(game, "traces")
    traced.reading = reading
    plain = planning_world(game, "uniform")
    agent = game.env.get_adhoc_agent()
    random.seed(seed)
    a = [traced.sample_state(agent) for _ in range(n)]
    random.seed(seed)
    b = [plain.sample_state(agent) for _ in range(n)]
    return a, b


def guessed(particle, name):
    return next(task for task in particle.components['tasks'] if task.index == name)


def test_rule2_hidden_task_lies_in_the_bearing_cone():
    game = make_game(steps=0)
    reading = {'teammates': {}, 'bearings': [((3, 3), (1.0, 0.0))], 'help': []}
    traced, _ = uniform_and_traces_particles(game, reading, n=200)
    cosines = [cosine((3, 3), guessed(p, 'H0').position, (1.0, 0.0)) for p in traced]
    assert min(cosines) > 0  # never behind or beside the emitter
    assert np.mean(cosines) > 0.8  # concentrated near the line


def test_rule2_bearing_at_a_known_task_adds_nothing():
    game = make_game(steps=0)
    reading = {'teammates': {}, 'bearings': [((2, 1), (1.0, 0.0))], 'help': []}  # points at T0 (4, 1), visible
    traced, plain = uniform_and_traces_particles(game, reading)
    for p, q in zip(traced, plain):  # same random numbers, same guesses as C1
        assert [(t.position, t.level) for t in p.components['tasks']] == \
               [(t.position, t.level) for t in q.components['tasks']]


def test_rule3_help_trace_yields_an_adjacent_task_with_level_L_plus_d():
    game = make_game(steps=0)
    reading = {'teammates': {}, 'bearings': [], 'help': [((6, 6), 1.25)]}
    traced, _ = uniform_and_traces_particles(game, reading, n=100)
    for p in traced:
        task = guessed(p, 'H0')
        assert task.position in [(7, 6), (5, 6), (6, 7), (6, 5)]
        assert task.level == pytest.approx(1.25)


def test_rule3_help_next_to_a_known_task_adds_nothing():
    game = make_game(steps=0)
    reading = {'teammates': {}, 'bearings': [], 'help': [((4, 2), 1.25)]}  # next to the visible T0 at (4, 1)
    traced, plain = uniform_and_traces_particles(game, reading)
    for p, q in zip(traced, plain):
        assert [(t.position, t.level) for t in p.components['tasks']] == \
               [(t.position, t.level) for t in q.components['tasks']]


def test_c2_reads_the_demo_scenario_correctly():
    """Evaluation scenario 0 after 8 idle steps (the breakpoint demo): w2 left the view heading East."""
    from src.utils.evaluation import create_evaluation_set
    scenario = create_evaluation_set(1, task_level_range=(0.3, 1.5))[0]
    random.seed(0)
    np.random.seed(0)
    game = StigmergicLevelForagingEnv(scenario.copy(), dim=DIM, trace_mode="intent")
    game.reset()
    for _ in range(8):
        game.step(4)
    truth = game.env.get_agent_by_index('w2').position
    world = planning_world(game, "traces")
    last = world.reading['teammates']['w2']
    assert last['cell'] == (5, 1) and last['k'] == 1 and last['radius'] == 1
    assert last['bearing'] == pytest.approx((1.0, 0.0))
    agent = game.env.get_adhoc_agent()
    hits = {}
    for belief in ("uniform", "traces"):
        world = planning_world(game, belief)
        random.seed(11)
        particles = [world.sample_state(agent) for _ in range(300)]
        hits[belief] = np.mean([p.get_agent_by_index('w2').position == truth for p in particles])
    assert hits["traces"] > 0.3 > 0.05 > hits["uniform"]
