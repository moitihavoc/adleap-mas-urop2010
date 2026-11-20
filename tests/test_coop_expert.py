import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.utils.coop_expert import choose_target, coop_expert_action

DIM = 10


def make_env(tasks, teammate_level=0.6, teammate_target=None, adhoc_pos=(1, 1), adhoc_direction=np.pi / 2):
    """Ad hoc agent A (level 0.9) and one teammate w1 at (4, 4), facing (3, 4)."""
    teammate = Agent(index='w1', atype='l1', position=(4, 4), direction=np.pi,
                     radius=1.0, angle=1.0, level=teammate_level)
    teammate.target = teammate_target
    components = {
        'agents': [Agent(index='A', atype='l1', position=adhoc_pos, direction=adhoc_direction,
                         radius=0.25, angle=1.0, level=0.9), teammate],
        'adhoc_agent_index': 'A',
        'template_types': ['l1'],
        'tasks': [Task(index=index, position=position, level=level) for index, position, level in tasks],
        'impostor_index': None,
        'obstacles': [],
    }
    return LevelForagingEnv(shape=(DIM, DIM), components=components, display=False,
                            type_knowledge=False, parameter_knowledge=True, vision_block=False)


def test_joins_a_teammate_it_can_complete_the_task_with():
    # T0 needs 1.2: teammate 0.6 + ad hoc 0.9 is enough. T1 is solo-able but 3 cells away (+3 solo penalty).
    env = make_env([('T0', (3, 4), 1.2), ('T1', (1, 4), 0.5)], teammate_target=(3, 4))
    assert choose_target(env).index == 'T0'


def test_skips_a_task_the_teammate_finishes_alone():
    env = make_env([('T0', (3, 4), 0.5), ('T1', (6, 7), 0.5)], teammate_level=0.8, teammate_target=(3, 4))
    assert choose_target(env).index == 'T1'


def test_goes_solo_when_there_is_nothing_to_join():
    # nobody targets T0, which the ad hoc agent cannot lift alone
    env = make_env([('T0', (3, 4), 1.2), ('T1', (6, 7), 0.5)])
    assert choose_target(env).index == 'T1'


def test_waits_at_the_nearest_task_when_nothing_is_feasible():
    env = make_env([('T0', (3, 4), 1.4), ('T1', (2, 2), 1.3)])
    assert choose_target(env).index == 'T1'


def test_presses_load_when_facing_its_target():
    # ad hoc agent at (2, 4) facing +x, i.e. facing T0 at (3, 4)
    env = make_env([('T0', (3, 4), 1.2)], teammate_target=(3, 4), adhoc_pos=(2, 4), adhoc_direction=0)
    assert coop_expert_action(env) == 4


def test_moves_towards_its_target_otherwise():
    env = make_env([('T0', (3, 4), 1.2)], teammate_target=(3, 4), adhoc_pos=(1, 4), adhoc_direction=np.pi / 2)
    assert coop_expert_action(env) == 0   # +x, towards (3, 4)
