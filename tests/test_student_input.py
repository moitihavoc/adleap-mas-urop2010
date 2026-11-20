import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.answer_key import NUM_ANSWER_KEY, answer_key_planes
from src.utils.evaluation import ADHOC_ANGLE, ADHOC_LEVEL, ADHOC_RADIUS, DIM
from src.utils.scenarios import make_scenario
from src.utils.tensor_convert import HEADINGS, NUM_SELF_FEATURES
from src.utils.train_cnn import accuracy

'''
=== Student input (P3, P5) and answer key (Gate 1) ===
'''


def make_env(seed):
    random.seed(seed)
    base_env = make_scenario(dim=DIM, nagents=2, ntasks=5, adhoc_level=ADHOC_LEVEL, adhoc_radius=ADHOC_RADIUS,
                             adhoc_angle=ADHOC_ANGLE, task_level_range=(0.3, 1.5))
    env = StigmergicLevelForagingEnv(base_env, dim=DIM, trace_mode="intent", self_features=True)
    return env, env.reset()


def test_input_has_15_channels():
    env, obs = make_env(0)
    assert obs.shape == (3 + 6 + 1 + NUM_SELF_FEATURES, DIM, DIM) == (15, DIM, DIM)


def test_own_position_and_heading():
    env, obs = make_env(1)
    for action in [0, 1, 2, 3, 0]:
        obs, _, done, _ = env.step(action)
        adhoc = env.env.get_adhoc_agent()
        position_plane = obs[10]
        assert position_plane.sum() == 1 and position_plane[adhoc.position[0], adhoc.position[1]] == 1
        heading = obs[11:15].mean(dim=(1, 2))           # each heading plane is all 0 or all 1
        assert heading.sum() == 1
        assert np.isclose(HEADINGS[int(heading.argmax())] % (2 * np.pi), adhoc.direction % (2 * np.pi))


def test_visible_levels_only():
    """Visible teammates/tasks carry their true level; anything outside the view is absent."""
    for seed in range(10):
        env, obs = make_env(seed)
        visible = env.env.get_observation()
        visible_tasks = {index for index, _, _ in visible['tasks']}
        visible_agents = {index for index, _, _ in visible['agents']}
        for task in env.env.components['tasks']:
            x, y = task.position
            expected = task.level if task.index in visible_tasks else 0.0
            assert np.isclose(obs[1, x, y].item(), expected)
        for agent in env.env.components['agents']:
            if agent.index == env.env.get_adhoc_agent().index:
                continue
            x, y = agent.position
            expected = agent.level if agent.index in visible_agents else 0.0
            assert np.isclose(obs[0, x, y].item(), expected)


def test_answer_key_holds_all_true_levels():
    env, _ = make_env(3)
    planes = answer_key_planes(env.env, DIM)
    assert planes.shape == (NUM_ANSWER_KEY, DIM, DIM)
    for task in env.env.components['tasks']:
        x, y = task.position
        assert np.isclose(planes[0, x, y].item(), 0.0 if task.completed else task.level)


def test_accuracy_on_known_scores():
    """The user's accuracy() from train_cnn.py, on a fake model with fixed action scores."""
    scores = torch.tensor([[0.1, 2.0, 0.3, 0.0, 0.5],
                           [1.5, 0.2, 0.1, 0.0, 0.4],
                           [0.0, 0.0, 0.0, 0.0, 9.0],
                           [0.0, 0.0, 3.0, 0.0, 0.0]])
    fake_model = lambda obs: (scores, None)
    assert accuracy(fake_model, None, torch.tensor([1, 4, 4, 0])) == 0.5
    assert accuracy(fake_model, None, torch.tensor([1, 0, 4, 2])) == 1.0
    assert accuracy(fake_model, None, torch.tensor([0, 1, 2, 3])) == 0.0
