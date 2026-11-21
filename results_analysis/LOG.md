# Lab notebook (branch `urop1020`)

One entry per experiment, oldest first. Every reported number comes from a script in this repo and a results file
named in the entry, so it can be rerun and checked. Older entries (POMCP rerun, teacher failures) live in
`results_analysis/LOG.md` on branch `post_urop1010`.

Entry template: **Question · Commit · Command(s) · Settings · Results · Caveats · Decision**

---

## 2026-11-06: Validation-split exercise (written by hand)

- **Question:** Does the original student overfit, and when does validation accuracy stop improving?
- **Commit:** `1887e72` (`src/utils/train_cnn.py`: episode-level 80/20 split, `accuracy()` after every epoch).
- **Command:** `python3 src/utils/train_cnn.py | tee results_analysis/Results/val_split_exercise.txt`
- **Settings:** `src/Training_Data/training_data_l1.pt` (300 L1 episodes, old pipeline), `random.seed(37)`, 240 training / 60 validation games (5,913 / 1,544 states), original `TraceCNN`, Adam 1e-3, batch 128, 30 epochs.
- **Results** (`results_analysis/Plots/val_split_exercise.png`): training accuracy 0.462 → 0.976; validation 0.432 → peak **0.605 at epoch 11** → 0.574 at epoch 30. Gap 0.03 → 0.40.
- **Caveats:** one seed and one split; validation noise is at least ±0.03 (steps within a game are correlated); old data (agent at level 1.0 with full view, L1 teacher).
- **Decision:** add early stopping on validation accuracy; fix the data before reading anything else into these numbers.

## 2026-11-06: Gate 0 fixes, test-first

- **Question:** Do the known setup bugs exist on `master`, and are they gone after the fixes?
- **Commit:** `1887e72` + uncommitted changes listed in `results_analysis/presentation_2026-11-06.md`.
- **Commands:** `python -m pytest tests -q`
- **Results:**
  - `tests/test_conditions.py` against `master`: **fails**, `assert 1.0 == 0.9` (the framework's `reset()` restores the snapshot taken at creation, so data collection used level 1.0 and full view). After `make_scenario`: passes.
  - Trace tests (`tests/test_traces.py`, `tests/test_diffusion.py`, ported from `post_urop1010`) against `master`'s `traces.py`: **10 of 12 fail** (`results_analysis/Results/gate0/traces_tests_before_fix.txt`), including diffusion copying only intensity / exceeding 1.0, age stuck at 1.0, standing help not refreshed. Some failures are only API differences (the old class lacks `emitter`, `num_fields`).
  - After porting the fixed `traces.py`, `StigmergicLevelForagingEnv.py`, `tensor_convert.py`, and adding `scenarios.py`, `coop_expert.py`, `answer_key.py`, own-position/heading/visible-level input: **32 of 32 pass** (`results_analysis/Results/gate0/tests_final.txt`).
- **Decision:** proceed to the Gate 1 measurement on the fixed setup.

## 2026-11-06: Gate 1 learning curve (answer-key positive control)

- **Question:** Can the student learn the cooperative teacher's rule, and is it limited by data or by missing information?
- **Commit:** `1887e72` + uncommitted Gate 0 changes.
- **Commands:**
  ```
  for w in 0 1 2 3; do python3 results_analysis/gate1_learning_curve.py --collect --episodes 525 --worker $w & done; wait
  python3 results_analysis/gate1_learning_curve.py --train --sizes 25 50 100 200 400 800 1600 --seeds 0 1
  ```
- **Settings:** hard tasks U(0.3, 1.5); ad hoc level 0.9, view radius 0.25 (same as evaluation); intent traces with local reading; cooperative teacher drives with 20 % random actions, every step labelled with the teacher's action. 2,100 episodes (37 timed out, 1.8 %), 69,139 states. 100 held-out episodes (3,007 states; majority-class accuracy 0.259), training sizes 25–1,600 episodes from the other 2,000, 2 seeds each. Inputs: student (15 channels) vs. student + answer key (18 channels). `TraceCNN` unchanged, policy loss only, Adam 1e-3, batch 128, early stopping (patience 5, max 40 epochs). Train accuracy is reported at the best-validation epoch.
- **Results** (`results_analysis/Results/gate1_learning_curve.csv`, `results_analysis/Plots/gate1_learning_curve.png`), held-out accuracy, mean of 2 seeds [min–max]:

  | Training episodes | States | Student | Student + answer key |
  |---|---|---|---|
  | 25 | 836 | 0.405 [0.397–0.413] | 0.417 [0.401–0.432] |
  | 50 | 1,551 | 0.450 [0.443–0.458] | 0.468 [0.467–0.469] |
  | 100 | 3,250 | 0.493 [0.484–0.502] | 0.495 [0.484–0.506] |
  | 200 | 6,900 | 0.586 [0.575–0.598] | 0.596 [0.590–0.602] |
  | 400 | 13,410 | 0.675 [0.652–0.698] | 0.701 [0.691–0.711] |
  | 800 | 26,504 | 0.743 [0.734–0.751] | 0.780 [0.772–0.789] |
  | 1,600 | 52,926 | **0.796** [0.794–0.797] | **0.841** [0.836–0.846] |

- **Reading:** both curves are still rising at the largest size (about +5–6 points per doubling), so **data is the main limit**. The answer key adds only ~4–5 points at the largest sizes, so missing information is a smaller limit. Gate 1 (≥ 0.90 with the answer key) is **not passed yet**.
- **Caveats:**
  - 2 seeds; a single held-out set of 100 episodes (3,007 states, correlated within episodes).
  - The held-out set also selects the early-stopping epoch, so it is a validation set and the accuracies are slightly optimistic; the next run should add a separate test set.
  - The October probe (47 % flat at 7,800 states) is not directly comparable: it changed several things at once relative to this run (its data was mostly student-driven; no visible levels; old trace code). At a similar size this run gets ~0.59.
  - "Data is the limit" is read from the slope; whether the curve keeps rising to 0.90 is an extrapolation until measured.
  - Accuracy measures copying the teacher, not playing well; no closed-loop games were run.
- **Decision:** next, (1) collect more data (≥ 4× this run) and extend the curve; (2) in parallel, test the corrected P2 idea (fewer position-specific weights: agent-centred input or no large dense layer) to see whether it reaches 0.90 with less data.

## 2026-11-06: Gate 1 student in real games vs. baselines

- **Question:** How does the student from the learning curve actually play, compared with the baselines and the teacher?
- **Commit:** `1887e72` + uncommitted Gate 0 changes.
- **Commands:**
  ```
  python3 results_analysis/evaluate_gate1.py --train-student
  python3 results_analysis/evaluate_gate1.py --evaluate
  python3 results_analysis/evaluate_gate1.py --plot
  ```
- **Settings:** student = `TraceCNN`, 15-channel input, trained on all 2,000 training-pool episodes (66,132 states) with early stopping on the 100 held-out episodes (held-out accuracy 0.803, epoch 7, seed 0). 100 hard evaluation scenarios (`create_evaluation_set(100, task_level_range=(0.3, 1.5))`, seed 2026); every episode seeds `random`, `np.random` and `torch` with its scenario index, as `evaluate_pomcp.py` on `post_urop1010` did. POMCP rows read from the logged `post_urop1010` CSVs, not rerun.
- **Results** (`results_analysis/Results/gate1_eval/*.csv`, `results_analysis/Plots/gate1_vs_baselines.png`), all tasks [95 % Wilson CI], mean steps:
  idle 39 % [30, 49] 133.8 · random 50 % [40, 60] 127.7 · L1 64 % [54, 73] 88.7 · **student top move 66 % [56, 75] 84.3 (34 timeouts)** · **student sampled 77 % [68, 84] 69.8 (23 timeouts)** · teacher 97 % [92, 99] 29.8 · POMCP own view 0.5 s 99 % · full info 0.5 s 100 % (logged).
  Paired, sampled student vs.: random 32 / 5 scenarios only-one-finished; L1 15 / 2; top-move student 17 / 6; teacher 3 / 23. The 3 are exactly the teacher's failures in this run (8, 53, 89).
- **Caveats:** one training seed. Idle/L1/teacher nearly but not exactly reproduce the 21 Oct log (L1 64 vs 63 %; teacher failures 8, 53, 89 here vs. 8, 53, 65 then; teacher steps 29.8 vs 30.3), so the runs are not bit-identical; cause not investigated. Sampling's gain is from escaping loops, not better decisions.
- **Decision:** the student is now clearly better than random and L1 when sampling, but deadlocks under argmax remain; keep sampling for Gate 2 comparisons, and fix the learner (Gate 1) before trace tests.

---

## 2026-11-20: New direction: POMCP that reads traces (instructor feedback of 2026-11-20)

Context: the instructor asked for MCTS + world models instead of model-free imitation: POMCP plans by simulating its world,
so with traces in that world it can be the teacher; the ad hoc agent must then *learn* what traces mean (traces as
observations). Plan: `~/.claude/plans/i-have-the-latest-functional-wadler.md`. Full method, math and source map:
`docs/trace_pomcp/trace_pomcp_method.pdf`. The Gate 1 BC student is shelved; `coop_expert` stays as a baseline only.

## 2026-11-20: Stage 2: does POMCP see traces? (breakpoint demo)

- **Question:** Do traces reach POMCP, and does it imagine them?
- **Commit:** `ab72bb9` + uncommitted (`src/envs/TracePlanningEnv.py`, `run_pomcp_episode` in `src/utils/evaluation.py`).
- **Command:** `python3 results_analysis/demo_pomcp_traces.py`
- **Results:** evaluation scenario 0, step 8 (ad hoc agent idle): teammate w2 is out of view but its traces are sensed at
  (3,1), (4,1), (5,1), ages 0.6/0.4/0.2, bearing East (truth: w2 at (6,1), heading to the unseen task at (8,1)). In the
  framework's world POMCP sees agents `['A']` only: **both teammates vanish** (a framework memory-check bug drops every
  invisible teammate, and the framework sampler never puts them back), and it guesses 6 tasks for 5 (U(0,1) levels).
  With `TracePlanningEnv` (belief "framework") one 0.1 s decision imagined 576 steps and 576 traces, all the ad hoc
  agent's own.
- **Decision:** carrying traces is not enough; the belief must read them (C1/C2).

## 2026-11-20: Stage 4: belief accuracy, C1 (no traces) vs C2 (reads traces)

- **Question:** Do POMCP's guesses of the hidden world get closer to the truth when the belief reads traces?
- **Commit:** `ab72bb9` + uncommitted (`read_traces`, Rules 1-3 in `src/envs/TracePlanningEnv.py`).
- **Command:** `python3 results_analysis/evaluate_trace_pomcp.py --belief` then `--summary`
  (figure: `python3 results_analysis/demo_pomcp_traces.py --figure`).
- **Settings:** 100 hard evaluation scenarios (seed 2026, U(0.3,1.5)), intent traces, local sensing, view 0.25; ad hoc
  driven by the cooperative teacher (only to visit realistic states); first 40 steps; 50 guesses per belief per decision,
  same observation and seed for C1 and C2. 95% CI clustered by scenario.
- **Results** (`results_analysis/Results/trace_pomcp/belief_accuracy.csv`, `results_analysis/Plots/pomcp_trace_belief.png`):

  | Hidden object | Cases | n | Distance C1 → C2 (cells) | Hit rate C1 → C2 |
  |---|---|---|---|---|
  | teammate | trace sensed | 810 (99 scen.) | 5.57 → **1.26** (−4.31 ± 0.20) | 0.019 → **0.428** (+0.41 ± 0.04) |
  | teammate | no trace sensed | 766 | 5.57 → 5.59 (+0.02 ± 0.03) | 0.017 → 0.016 |
  | teammate | all | 1,576 | 5.57 → 3.36 (−2.21 ± 0.20) | 0.018 → 0.228 |
  | task | bearing/help sensed | 2,646 | 3.20 → 2.90 (−0.30 ± 0.11) | 0.081 → 0.130 (+0.05 ± 0.02) |
  | task | none sensed | 529 | 2.29 → 2.29 | 0.084 → 0.084 |

  Demo moment (2,000 guesses): w2 on its true cell 1.5 % (C1) → 98.8 % (C2); guessed tasks on w2's true target (8,1)
  0.09 → 0.23.
- **Caveats:** teacher-driven states, not POMCP-driven; "no trace" rows are a sanity check (C2 must equal C1 there; it
  does up to random-number order). Rule constants (weight 1+2cos, cone exponent 4) were not tuned. C1 ignores where the
  agent last saw a teammate, so this is traces vs. no information, not traces vs. memory.
- **Decision:** traces carry a lot of information about hidden teammates and some about hidden tasks; next, check whether
  it changes play (games).

## 2026-11-20: Stage 5: games, POMCP C0 / C1 / C2

- **Question:** Does POMCP play better when its belief reads traces (C2) than when it does not (C1)?
- **Commit:** `ab72bb9` + uncommitted (`src/envs/TracePlanningEnv.py`, `run_pomcp_episode`, `evaluate_trace_pomcp.py`).
- **Commands:**
  ```
  for w in $(seq 0 17); do python3 results_analysis/evaluate_trace_pomcp.py --games \
      --configs C1-0.1 C2-0.1 C0-0.1 C1-0.5 C2-0.5 C0-0.5 --worker $w --num-workers 18 & done; wait
  for w in $(seq 0 17); do python3 results_analysis/evaluate_trace_pomcp.py --games --configs C1-2.0 C2-2.0 \
      --worker $w --num-workers 18 & done; wait
  python3 results_analysis/evaluate_trace_pomcp.py --summary
  ```
- **Settings:** 100 hard evaluation scenarios (seed 2026, U(0.3,1.5)), intent traces, own view 0.25, 200-step cap,
  per-scenario seeding. POMCP: depth 20, K = 100 particles, c = 0.5, γ = 0.95. C0 = framework world with tree reuse
  (October set-up); C1/C2 = `TracePlanningEnv`, new tree every decision. 18 parallel workers (OMP_NUM_THREADS=1).
- **Results** (`results_analysis/Results/trace_pomcp/games/*.csv`):

  | Config | Completed [95% CI] | Steps | Steps if done | Timeouts | Tree steps/dec. | Median s/dec. |
  |---|---|---|---|---|---|---|
  | C0-0.1 | 89% [81, 94] | 70.4 | 54.4 | 11 | 24 | 0.12 |
  | C1-0.1 | 79% [70, 86] | 87.1 | 57.1 | 21 | 8 | – |
  | C2-0.1 | 82% [73, 88] | 86.0 | 61.0 | 18 | 7 | – |
  | C0-0.5 | 95% [89, 98] | 43.4 | 35.2 | 5 | 203 | 0.53 |
  | C1-0.5 | 89% [81, 94] | 66.0 | 49.5 | 11 | 59 | 0.54 |
  | C2-0.5 | 92% [85, 96] | 71.1 | 59.9 | 8 | 68 | 0.54 |
  | C1-2.0 | 94% [88, 97] | 55.2 | 46.0 | 6 | 266 | 2.05 |
  | C2-2.0 | 97% [92, 99] | 54.1 | 49.6 | 3 | 363 | 2.05 |

  Paired C2 − C1 steps (all scenarios / both finish): 0.1 s −1.1 ± 11.2 / −0.6 ± 8.4 (n = 73);
  0.5 s +5.0 ± 9.4 / +7.6 ± 8.0 (n = 86); 2.0 s −1.1 ± 8.7 / +3.9 ± 6.3 (n = 91). Only C2 / only C1 finishes:
  9/6, 6/3, 6/3.
- **Reading:** no detectable trace effect on play at any budget, despite the large belief gain (previous entry). Not only
  a search problem: at 2.0 s C1/C2 search more than C0-0.5 and still show no effect, and still play worse than C0.
  Hypothesis (untested): POMCP's teammate model (an untyped teammate gets a random l1–l6 heuristic in imagination) is
  the bottleneck, so knowing *where* teammates are does not help predict *what they do*.
- **Caveats:**
  - The laptop suspended (lid closed) before 13:01 and from 13:06 to 14:15. Outcomes are valid (processes paused); mean
    seconds per decision are inflated, so medians are reported.
  - Each worker ran C1 before C2, so C2 often ran with less CPU contention and got more search (363 vs 266 tree steps at
    2.0 s). This bias favours C2, yet no effect appeared.
  - C1/C2 wall time per decision also includes drawing 100 fresh particles outside the search timer.
  - One game per scenario × config.
  - The 2.0 s budget was added during the day (not in the plan).
  - Two C2-2.0 workers stopped without an error message (scenarios 90 and 93); those two games were rerun alone
    (`--worker 0` / `--worker 3`).
- **Decision:** next, the teammate-model test (C1/C2 with true teammate types in imagination). Then C3, a learned trace
  meaning, scored by the share of the C1 → C2 gap it closes (on belief accuracy, and on play if a gap appears).
