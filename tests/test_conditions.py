import os
import random
import sys

sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import generate_random_estimation_scenario
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.utils.evaluation import ADHOC_LEVEL, ADHOC_RADIUS, DIM, create_evaluation_set

'''
=== Training and test conditions ===
The ad hoc agent must have the same level and field of view when the training data is collected
as when the student is evaluated; otherwise the student is tested on a distribution it never saw.
'''


def test_training_agent_matches_evaluation():
    """Build the environment the way src/utils/collect_data.py does (lines 25-43) and check the agent after reset."""
    random.seed(0)
    base_env, _ = generate_random_estimation_scenario(
        method='l1', adhoc_pos=(1, 1), dim=(DIM, DIM), nagents=2, ntasks=5,
        type_knowledge=False, parameter_knowledge=True, vision_block=False,
        template_types=['l1', 'l2'], parameters_minmax=[0.5, 1.0], seed=None, display=False
    )
    base_env.get_adhoc_agent().level = 0.9
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
