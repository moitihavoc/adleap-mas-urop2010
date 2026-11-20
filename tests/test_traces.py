import importlib.util
import os

import numpy as np

# load traces.py directly so the test needs only numpy (src/__init__ pulls in heavier deps)
_path = os.path.join(os.path.dirname(__file__), "..", "src", "communication", "traces.py")
_spec = importlib.util.spec_from_file_location("traces", _path)
_traces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_traces)
TraceField = _traces.TraceField

INTENSITY, AGE, LEVEL, HELP, CLAIM = range(5)
NEIGHBORS = [(1, 2), (3, 2), (2, 1), (2, 3)]


def test_diffuse_spreads_whole_trace():
    t = TraceField(dim=5)
    t.fusion(level=0.9, help=1, payload=0, position=(2, 2))
    t.diffuse()
    for x, y in NEIGHBORS:
        assert np.isclose(t.fields[INTENSITY, x, y], 0.3)
        assert t.fields[LEVEL, x, y] == np.float32(0.9)
        assert t.fields[HELP, x, y] == 1
        assert t.fields[CLAIM, x, y] == 0
        assert t.fields[AGE, x, y] == t.fields[AGE, 2, 2]
    assert t.fields[INTENSITY, 0, 0] == 0  # non-neighbors untouched


def test_diffuse_does_not_overwrite_emitters_and_stays_bounded():
    t = TraceField(dim=5)
    t.emit(0.9, 0, 1, (2, 2))
    t.emit(0.5, 1, 0, (2, 3))
    t.diffuse()
    assert t.fields[INTENSITY].max() <= 1.0
    assert t.fields[HELP, 2, 2] == 0 and t.fields[LEVEL, 2, 2] == np.float32(0.9)
    assert t.fields[HELP, 2, 3] == 1 and t.fields[LEVEL, 2, 3] == np.float32(0.5)


def test_overlapping_diffusion_prefers_help():
    t = TraceField(dim=5)
    t.emit(0.9, 0, 1, (2, 1))  # claim emitter
    t.emit(0.5, 1, 0, (2, 3))  # help emitter, both spread onto (2, 2)
    t.diffuse()
    assert t.fields[HELP, 2, 2] == 1
    assert t.fields[LEVEL, 2, 2] == np.float32(0.5)


def test_diffused_copies_do_not_spread_further():
    t = TraceField(dim=7)
    t.emit(0.9, 1, 0, (3, 3))
    t.diffuse()
    t.diffuse()
    assert t.fields[INTENSITY, 3, 5] == 0  # two cells away stays empty


def test_standing_help_trace_is_refreshed():
    t = TraceField(dim=5)
    t.fusion(0.5, 1, 0, (1, 1))
    for _ in range(10):
        t.decay()
        t.fusion(0.5, 1, 0, (1, 1))
        assert t.fields[INTENSITY, 1, 1] == 1.0
        assert t.fields[HELP, 1, 1] == 1


def test_help_not_overwritten_by_non_help():
    t = TraceField(dim=5)
    t.emit(0.5, 1, 0, (1, 1))
    t.decay()
    t.fusion(0.9, 0, 1, (1, 1))
    assert t.fields[HELP, 1, 1] == 1


def test_decay_reaches_exact_zero_and_clears_all_channels():
    t = TraceField(dim=5)
    t.emit(0.5, 1, 0, (1, 1))
    for _ in range(5):
        t.decay()
    assert not t.fields.any()
    assert not t.emitter.any()


def test_age_grows_from_zero():
    t = TraceField(dim=5)
    t.emit(0.5, 0, 1, (1, 1))
    assert t.fields[AGE, 1, 1] == 0
    t.decay()
    assert np.isclose(t.fields[AGE, 1, 1], 0.2)


def test_reset_clears_everything():
    t = TraceField(dim=5)
    t.emit(0.5, 1, 0, (1, 1))
    t.diffuse()
    t.reset()
    assert not t.fields.any() and not t.emitter.any()


def test_intent_payload_spreads_with_continuous_help():
    t = TraceField(dim=5, num_fields=6)  # intensity, age, level, help, bearing_x, bearing_y
    t.fusion(level=0.6, help=0.6, payload=(-1.0, 0.0), position=(2, 2))
    t.diffuse()
    for x, y in NEIGHBORS:
        assert np.isclose(t.fields[INTENSITY, x, y], 0.3)
        assert np.isclose(t.fields[HELP, x, y], 0.6)
        assert (t.fields[4, x, y], t.fields[5, x, y]) == (-1.0, 0.0)


def test_small_help_deficit_still_has_priority():
    t = TraceField(dim=5, num_fields=6)
    t.emit(0.9, 0.0, (0.0, 1.0), (2, 1))   # no help
    t.emit(0.5, 0.1, (1.0, 0.0), (2, 3))   # small deficit, both spread onto (2, 2)
    t.diffuse()
    assert np.isclose(t.fields[HELP, 2, 2], 0.1)
    assert np.isclose(t.fields[LEVEL, 2, 2], 0.5)
    t.decay()
    t.fusion(0.9, 0.0, (0.0, 1.0), (2, 2))  # an agent without help steps onto the help copy
    assert np.isclose(t.fields[HELP, 2, 2], 0.1)
