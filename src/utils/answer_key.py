import torch

from src.utils.coop_expert import committed_level

'''
=== Answer key (positive control for Gate 1) ===
The hidden information the cooperative teacher (src/utils/coop_expert.py) reads and the student does not see.
Appended to the student input only in the "student + answer key" condition: with it and the student's own
position/heading planes, the teacher's choice is a fixed function of the input, so a student that cannot learn it
from this input has a learning problem, not a missing-information problem.
    plane 0: level of every unfinished task, in or out of view
    plane 1: level of every teammate at its cell, in or out of view
    plane 2: total level of the teammates committed to each unfinished task, at the task's cell (their targets)
'''

NUM_ANSWER_KEY = 3


def answer_key_planes(env, dim):
    """Answer-key planes (NUM_ANSWER_KEY, dim, dim) read from the real LevelForagingEnv."""
    planes = torch.zeros((NUM_ANSWER_KEY, dim, dim), dtype=torch.float32)
    adhoc = env.get_adhoc_agent()
    teammates = [agent for agent in env.components['agents'] if agent.index != adhoc.index]

    for task in env.components['tasks']:
        if task.completed:
            continue
        x, y = task.position
        planes[0, x, y] = task.level
        planes[2, x, y] = committed_level(task, teammates)

    for teammate in teammates:
        x, y = teammate.position
        planes[1, x, y] = teammate.level

    return planes
