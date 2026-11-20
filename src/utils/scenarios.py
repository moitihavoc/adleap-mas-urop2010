import random

from src.envs.LevelForagingEnv import generate_random_estimation_scenario


def make_scenario(dim=10, nagents=2, ntasks=5, adhoc_pos=(1, 1), adhoc_level=0.9,
                  adhoc_radius=1.0, adhoc_angle=1.0, task_level_range=(0.0, 1.0),
                  template_types=('l1', 'l2'), parameters_minmax=(0.5, 1.0)):
    """
    Build a random Level-Based Foraging scenario with explicit ad hoc agent and task settings.

    Wraps the framework's generate_random_estimation_scenario (which hard-codes the ad hoc agent to
    level 1.0 and task levels to U(0, 1)) without modifying it.
        adhoc_level, adhoc_radius, adhoc_angle: ad hoc agent parameters (1.0 radius/angle = full observability)
        task_level_range: tasks are drawn from U(low, high); anything above the best single agent's level
            can only be completed cooperatively
        parameters_minmax: range of the teammates' level, radius and angle
    """
    base_env, _ = generate_random_estimation_scenario(
        method='l1', adhoc_pos=adhoc_pos, dim=(dim, dim),
        nagents=nagents, ntasks=ntasks, type_knowledge=False,
        parameter_knowledge=True, vision_block=False,
        template_types=list(template_types), parameters_minmax=list(parameters_minmax),
        seed=None, display=False
    )

    base_env.reset()  # initialises the attributes copy() relies on
    adhoc_agent = base_env.get_adhoc_agent()
    adhoc_agent.level = adhoc_level
    adhoc_agent.radius = adhoc_radius
    adhoc_agent.angle = adhoc_angle

    # the generator already drew U(0, 1); only redraw (and consume RNG) when another range is requested
    if tuple(task_level_range) != (0.0, 1.0):
        low, high = task_level_range
        for task in base_env.components['tasks']:
            task.level = random.uniform(low, high)

    # LevelForagingEnv.reset() restores the components the env was built with, which would silently undo
    # the changes above. copy() rebuilds the env from the current components, so they become the initial ones.
    env = base_env.copy()
    env.reset()
    return env
