import numpy as np

# Trace components stored per cell, for each trace mode (see StigmergicLevelForagingEnv).
# The first four are shared; the remaining ones are the mode's payload.
TRACE_FIELDS = {
    "original": ("intensity", "age", "level", "help", "claim"),
    "intent": ("intensity", "age", "level", "help", "bearing_x", "bearing_y"),
}


class TraceField:
    """
    Trace Field Class: Manage fields of trace components.
    Components:
            intensity: float - how much attention should we give to the signal
            age: how many steps ago was this trace created
            level: the level of the agent creating this signal
            help: > 0 when the emitter requires help
                  (original mode: 0 or 1; intent mode: level deficit measured on a failed load)
            payload: the remaining components of the trace mode
                  (original mode: claim signal, 0 or 1; intent mode: bearing to the emitter's target, x and y)
    Methods:
        decay: decay the intensity and age of all traces in the field
        diffuse: spread every emitted trace (all components) to its 4 neighboring cells
        emit: emit a trace in the field
        fusion: resolve overlapping traces in the field
        reset: clear all traces
    """

    def __init__(self, dim: int, decay_rate:float = 0.2, diffusion_rate: float = 0.3, num_fields:int = 5):
        self.dim = dim
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.fields = np.zeros((num_fields, dim, dim), dtype=np.float32)  # intensity, age, level, help, payload...
        self.emitter = np.zeros((dim, dim), dtype=bool)  # cells holding a trace emitted by an agent (not a diffused copy)

    def decay(self):
        self.fields[0] = np.maximum(0.0, self.fields[0] - self.decay_rate)  # Decay intensity
        self.fields[1] = np.minimum(self.fields[1] + 0.2, 1.0)  # Increment age
        zero_mask = self.fields[0] <= 1e-6 # remove traces which has (numerically) 0 intensity
        for ch in range(self.fields.shape[0]):
            self.fields[ch][zero_mask] = 0.0
        self.emitter[zero_mask] = False

    def diffuse(self):
        """
        Spread every trace emitted by an agent to its 4 neighboring cells.
        A neighbor receives the whole trace (age, level, help, payload) from its source,
        with intensity scaled by diffusion_rate. Only emitted traces spread (not already
        diffused copies), cells holding an emitted trace are never overwritten, and
        overlapping spreads resolve with help signal > intensity (as in fusion).
        """
        src_mask = self.emitter & (self.fields[0] > 0)
        snapshot = self.fields.copy()
        d = self.dim

        # (source slice, destination slice) for shifting by one cell in each direction
        shifts = [
            ((slice(0, d - 1), slice(None)), (slice(1, d), slice(None))),      # down
            ((slice(1, d), slice(None)), (slice(0, d - 1), slice(None))),      # up
            ((slice(None), slice(0, d - 1)), (slice(None), slice(1, d))),      # right
            ((slice(None), slice(1, d)), (slice(None), slice(0, d - 1))),      # left
        ]

        for src, dst in shifts:
            cand_mask = src_mask[src]
            cand_intensity = snapshot[0][src] * self.diffusion_rate
            cand_score = cand_intensity + 2.0 * (snapshot[3][src] > 0)
            dst_score = self.fields[0][dst] + 2.0 * (self.fields[3][dst] > 0)
            write = cand_mask & ~self.emitter[dst] & (cand_score > dst_score)
            if not write.any():
                continue
            self.fields[0][dst][write] = cand_intensity[write]
            for ch in range(1, self.fields.shape[0]):
                self.fields[ch][dst][write] = snapshot[ch][src][write]

    def emit(self, level:float, help:float, payload, position:tuple):
        """
        Emit a trace in the field at the given position.
        payload: value(s) for the components after help (a scalar claim, or the bearing (x, y)).
        """
        x, y = position
        self.fields[0, x, y] = 1.0
        self.fields[1, x, y] = 0.0  # age starts at 0 and grows in decay()
        self.fields[2, x, y] = level
        self.fields[3, x, y] = help
        self.fields[4:, x, y] = payload
        self.emitter[x, y] = True

    def fusion(self, level:float, help:float, payload, position:tuple, intensity:float = 1.0):
        """
        Resolve overlapping traces in the field.
        Priority: help signal > intensity.
        """
        x, y = position
        if help > 0:
            self.emit(level, help, payload, position)  # help signal always wins (and refreshes a standing one)
        elif self.fields[3, x, y] > 0:
            pass
        elif self.fields[0, x, y] < intensity:
            self.emit(level, help, payload, position)

    def copy(self):
        """An independent copy (used when POMCP clones its imagined world, src/envs/TracePlanningEnv.py)."""
        other = TraceField.__new__(TraceField)
        other.dim, other.decay_rate, other.diffusion_rate = self.dim, self.decay_rate, self.diffusion_rate
        other.fields = self.fields.copy()
        other.emitter = self.emitter.copy()
        return other

    def reset(self):
        self.fields.fill(0.0)  # Clear all traces
        self.emitter.fill(False)
