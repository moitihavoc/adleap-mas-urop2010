import os
import random
import sys

sys.path.append(os.getcwd())

from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.evaluation import ADHOC_ANGLE, ADHOC_LEVEL, ADHOC_RADIUS, DIM, create_evaluation_set
from src.utils.scenarios import make_scenario

'''
=== Training and test conditions ===
The ad hoc agent must have the same level and field of view when the training data is collected
as when the student is evaluated; otherwise the student is tested on a distribution it never saw.
'''


def test_training_agent_matches_evaluation():
    """Build the environment the way src/utils/collect_data.py does (make_scenario) and check the agent after reset."""
    random.seed(0)
    base_env = make_scenario(dim=DIM, nagents=2, ntasks=5, adhoc_level=ADHOC_LEVEL,
                             adhoc_radius=ADHOC_RADIUS, adhoc_angle=ADHOC_ANGLE)
    env = StigmergicLevelForagingEnv(base_env, dim=DIM)
    env.reset()

    adhoc_agent = env.env.get_adhoc_agent()
    assert adhoc_agent.level == ADHOC_LEVEL
    assert adhoc_agent.radius == ADHOC_RADIUS


def test_evaluation_agent_keeps_its_settings():
    """Build the environment the way src/utils/evaluation.py run_evaluation does and check the agent after reset."""
    base_env = create_evaluation_set(num_scenarios=1)[0]
    env = StigmergicLevelForagingEnv(base_env.copy(), dim=DIM)
    env.reset()

    adhoc_agent = env.env.get_adhoc_agent()
    assert adhoc_agent.level == ADHOC_LEVEL
    assert adhoc_agent.radius == ADHOC_RADIUS
