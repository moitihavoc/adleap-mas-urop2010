"""
Visual check of TraceField diffusion (not meant to be committed).

Run:  .venv/bin/python tests/test_diffusion.py

Two scripted agents move around a small grid. Each step follows the same order as
StigmergicLevelForagingEnv.step: decay -> fusion (each agent emits) -> diffuse.
After every step the five trace channels are printed side by side.
Legend: 'A' / 'B' mark the agents' current cells, '.' means empty.
"""
import importlib.util
import os

import numpy as np

_path = os.path.join(os.path.dirname(__file__), "..", "src", "communication", "traces.py")
_spec = importlib.util.spec_from_file_location("traces", _path)
_traces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_traces)
TraceField = _traces.TraceField

DIM = 7
CHANNELS = ["intensity", "age", "level", "help", "claim"]

# (name, level, help, claim, path). Agent A asks for help, agent B can claim a task.
AGENTS = [
    ("A", 0.4, 1, 0, [(1, 1), (1, 2), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3)]),
    ("B", 0.9, 0, 1, [(5, 5), (5, 4), (4, 4), (3, 4), (2, 4), (2, 3), (2, 3), (2, 3)]),
]


def render(traces, agent_cells):
    """Return the printable lines of all channels side by side."""
    blocks = []
    for ch, name in enumerate(CHANNELS):
        lines = [name.center(DIM * 5)]
        for x in range(DIM):
            row = []
            for y in range(DIM):
                tag = agent_cells.get((x, y), " ")
                v = traces.fields[ch, x, y]
                cell = "  .  " if v == 0 else f"{v:4.2f}"
                if v == 0:
                    cell = f" .{tag} " if tag != " " else "  .  "
                else:
                    cell = f"{v:4.2f}{tag}" if tag != " " else f"{v:4.2f} "
                row.append(cell)
            lines.append("".join(row))
        blocks.append(lines)
    return ["   |   ".join(parts) for parts in zip(*blocks)]


def main():
    traces = TraceField(dim=DIM)
    steps = len(AGENTS[0][4])

    for t in range(steps):
        traces.decay()
        agent_cells = {}
        for name, level, help_, claim, path in AGENTS:
            pos = path[t]
            agent_cells[pos] = name
            traces.fusion(level, help_, claim, pos)
        traces.diffuse()

        moves = ", ".join(f"{name}@{path[t]}" for name, _, _, _, path in AGENTS)
        print(f"\n=== step {t}: {moves} ===")
        print("\n".join(render(traces, agent_cells)))
        print(f"max intensity={traces.fields[0].max():.2f}  "
              f"trace cells={int((traces.fields[0] > 0).sum())}  "
              f"emitter cells={int(traces.emitter.sum())}")


def test_diffusion_visual():
    """Pytest entry point: just runs the printout (use `pytest -s` to see it)."""
    main()


if __name__ == "__main__":
    main()
