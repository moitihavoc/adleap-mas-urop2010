import numpy as np
import torch

def dict_to_tensor(obs_dict:dict, fields:np.ndarray, dim:int, visible_grids:list, local_traces:bool = False):
    """
    Combines the original LBF env's observation dictionary with the trace field into a tensor representation for CNN training.
    The tensor has 3 + (number of trace fields) + 1 channels; with the original trace mode (5 fields) it has 9:
    - channel 1: agent positions (presence of agents in the grid)
    - channel 2: task positions (presence of tasks in the grid)
    - channel 3: obstacle positions (if any)
    - channel 4: trace intensity
    - channel 5: trace age
    - channel 6: trace agent-level
    - channel 7: trace help signal
    - channel 8: trace claim signal (intent mode: channels 8-9 hold the bearing x, y instead)
    - last channel: visibility mask (1 if visible, 0 otherwise)
    local_traces: the agent can only sense traces inside its visibility mask (trace channels are 0 elsewhere)
    """
    num_fields = fields.shape[0]
    tensor = np.zeros((3 + num_fields + 1, dim, dim), dtype=np.float32)

    for agent in obs_dict['agents']:
        x, y = agent[1],agent[2]
        tensor[0, x, y] = 1.0

    for task in obs_dict['tasks']:
        x, y = (task[1], task[2])
        tensor[1, x, y] = 1.0

    if (obs_dict['obstacles']):
        for obstacle in obs_dict['obstacles']:
            x, y = obstacle.position
            tensor[2, x, y] = 1.0

    # Fill in the trace field channels
    tensor[3:3 + num_fields, :, :] = fields

    # apply visibility mask
    for (x, y) in visible_grids:
        tensor[-1, x, y] = 1.0

    if local_traces:
        tensor[3:3 + num_fields, :, :] *= tensor[-1]

    return torch.from_numpy(tensor).float()


HEADINGS = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)  # agent.direction values, in the order of the heading planes
NUM_SELF_FEATURES = 1 + len(HEADINGS)


def add_self_features(obs_tensor:torch.Tensor, obs_dict:dict, env):
    """
    Student input with the information the ad hoc agent legitimately has about itself and what it sees (fixes P3, P5):
    - channel 1 (teammates) holds each visible teammate's level instead of 1 (the scenarios use parameter_knowledge=True)
    - channel 2 (tasks) holds each visible unfinished task's level instead of 1 (tasks outside the view stay unknown)
    - appended: 1 own-position plane (1 at the agent's cell) and 4 heading planes (one-hot, the whole plane is 1)
    Returns a tensor with NUM_SELF_FEATURES more channels than obs_tensor.
    """
    out = obs_tensor.clone()
    agent_levels = {agent.index: agent.level for agent in env.components['agents']}
    task_levels = {task.index: task.level for task in env.components['tasks']}
    for index, x, y in obs_dict['agents']:
        out[0, x, y] = agent_levels[index]
    for index, x, y in obs_dict['tasks']:
        out[1, x, y] = task_levels[index]

    adhoc = env.get_adhoc_agent()
    dim = obs_tensor.shape[-1]
    extra = torch.zeros((NUM_SELF_FEATURES, dim, dim), dtype=out.dtype)
    extra[0, adhoc.position[0], adhoc.position[1]] = 1.0
    heading = int(np.argmin([abs(np.angle(np.exp(1j * (adhoc.direction - h)))) for h in HEADINGS]))
    extra[1 + heading] = 1.0
    return torch.cat([out, extra])


def permute_trace_cells(obs:torch.Tensor, permutation:torch.Tensor = None, generator:torch.Generator = None):
    """
    Spatially shuffle the trace channels of one observation (C, H, W); every other channel stays in place.
    With full visibility the given `permutation` of all cells is applied (so a whole episode can share one).
    Under local reading (some cells invisible) the trace values are permuted among the visible cells only,
    so they stay inside the area the agent can sense, as real traces do.
    """
    num_trace = obs.shape[0] - 4
    spatial_size = obs.shape[-2] * obs.shape[-1]
    out = obs.clone()
    traces = obs[3:3 + num_trace].reshape(num_trace, spatial_size)
    visible = obs[-1].reshape(-1) > 0

    if visible.all():
        if permutation is None:
            permutation = torch.randperm(spatial_size, generator=generator)
        shuffled = traces[:, permutation]
    else:
        cells = visible.nonzero().squeeze(1)
        shuffled = torch.zeros_like(traces)
        shuffled[:, cells] = traces[:, cells[torch.randperm(len(cells), generator=generator)]]

    out[3:3 + num_trace] = shuffled.reshape(num_trace, obs.shape[-2], obs.shape[-1])
    return out
