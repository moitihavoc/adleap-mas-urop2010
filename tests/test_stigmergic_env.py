import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv

DIM = 10
# channels of the intent-mode observation
INTENSITY, AGE, LEVEL, HELP, BEARING_X, BEARING_Y, VISIBLE = 3, 4, 5, 6, 7, 8, 9


def make_env(adhoc_pos, trace_mode="intent", **kwargs):
    """
    Teammate w1 (level 0.6) stands at (4, 4) facing task T0 (level 1.2) at (3, 4).
    Being an L1 agent that sees T0, it presses load on the first step and fails.
    The ad hoc agent A (level 0.9) has a 0.25 radius view (about 3.5 cells).
    """
    components = {
        'agents': [
            Agent(index='A', atype='l1', position=adhoc_pos, direction=np.pi / 2,
                  radius=0.25, angle=1.0, level=0.9),
            Agent(index='w1', atype='l1', position=(4, 4), direction=np.pi,
                  radius=1.0, angle=1.0, level=0.6),
        ],
        'adhoc_agent_index': 'A',
        'template_types': ['l1'],
        'tasks': [Task(index='T0', position=(3, 4), level=1.2)],
        'impostor_index': None,
        'obstacles': [],
    }
    base_env = LevelForagingEnv(shape=(DIM, DIM), components=components, display=False,
                                type_knowledge=False, parameter_knowledge=True, vision_block=False)
    env = StigmergicLevelForagingEnv(base_env, dim=DIM, trace_mode=trace_mode, **kwargs)
    env.reset()
    return env


def test_failed_load_writes_help_deficit_and_bearing():
    env = make_env(adhoc_pos=(3, 1))
    obs, _, _, _ = env.step(4)  # the ad hoc agent stays put

    assert obs.shape == (10, DIM, DIM)
    assert obs[HELP, 4, 4] == pytest.approx(0.6)          # 1.2 - 0.6
    assert obs[LEVEL, 4, 4] == pytest.approx(0.6)
    assert (obs[BEARING_X, 4, 4], obs[BEARING_Y, 4, 4]) == (-1.0, 0.0)  # towards T0
    # the whole mark diffuses to a visible neighbour
    assert obs[INTENSITY, 4, 3] == pytest.approx(0.3)
    assert obs[HELP, 4, 3] == pytest.approx(0.6)


def test_adhoc_mark_has_no_bearing_and_no_help_without_failed_load():
    env = make_env(adhoc_pos=(3, 1))
    obs, _, _, _ = env.step(4)  # A faces (3, 2), where there is no task: not a failed load

    assert obs[INTENSITY, 3, 1] == 1.0
    assert obs[LEVEL, 3, 1] == pytest.approx(0.9)
    assert obs[HELP, 3, 1] == 0.0
    assert (obs[BEARING_X, 3, 1], obs[BEARING_Y, 3, 1]) == (0.0, 0.0)


def test_help_mark_is_refreshed_while_the_teammate_keeps_failing():
    env = make_env(adhoc_pos=(3, 1))
    for _ in range(8):
        obs, _, _, _ = env.step(4)
    assert obs[INTENSITY, 4, 4] == 1.0
    assert obs[HELP, 4, 4] == pytest.approx(0.6)


def test_local_reading_hides_traces_outside_the_view():
    env = make_env(adhoc_pos=(8, 8))  # (4, 4) is about 5.7 cells away
    obs, _, _, _ = env.step(4)

    assert env.traces.fields[3, 4, 4] == pytest.approx(0.6)   # the mark exists in the environment
    assert obs[VISIBLE, 4, 4] == 0.0
    assert not obs[INTENSITY:VISIBLE, 4, 4].any()              # but the ad hoc agent cannot sense it
    assert obs[INTENSITY, 8, 8] == 1.0                         # its own cell is visible


def test_global_reading_can_be_requested():
    env = make_env(adhoc_pos=(8, 8), local_traces=False)
    obs, _, _, _ = env.step(4)
    assert obs[HELP, 4, 4] == pytest.approx(0.6)


def test_original_mode_keeps_the_report_layout():
    env = make_env(adhoc_pos=(3, 1), trace_mode="original")
    obs, _, _, _ = env.step(4)
    assert obs.shape == (9, DIM, DIM)
    assert env.observation_space.shape == (9, DIM, DIM)
    assert env.local_traces is False


def test_unknown_trace_mode_is_rejected():
    with pytest.raises(ValueError):
        make_env(adhoc_pos=(3, 1), trace_mode="target_cell")
