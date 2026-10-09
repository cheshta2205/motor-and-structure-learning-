# Brief: the motor-and-structure-learning project, for manuscript and figure planning

This is a handoff summary of a computational neuroscience project, written so that an assistant can help plan the manuscript's structure and its high-level figure designs. Section 1 is what exists (model, tasks, experiments, results). Section 2 is what the author wants to do next. Numbers are from runs completed on 2026-09-21/22; ten seeds per configuration unless stated.

## 1. What the project contains

### 1.1 The model: a three-module circuit

- **CB (cerebellum)**: a forward model that predicts the next state from the current state, the action and a context latent: `s_hat_t = s_in + CB([s_in, a_t, z_t])`.
- **EC / MTL (entorhinal cortex, medial temporal lobe)**: a recurrent network (GRU) that infers a context latent `z_t` from the cerebellar prediction error alone: `e_t = s_t - s_hat_t`, `z_t, h_t = EC([s_{t-1}, a_{t-1}, e_t], h_{t-1})`. In the main arm EC never sees the current state directly; the prediction error is its only window on the present. A control arm additionally gives EC the current state.
- **MC (motor cortex, the controller)**: a soft actor-critic reinforcement-learning agent that receives the kinematic state, the goal, and the circuit's latent `z` (plus a projection of CB's hidden state), and produces 2-D forces.

The circuit (EC and CB) is trained by supervised next-state prediction on sequences with hazard-switching contexts, then frozen. MC is trained on top of the frozen circuit. The claim is that a cerebellar prediction-error signal to the medial temporal lobe is what lets a motor context be inferred and represented.

### 1.2 The task

A point mass reaching in a 2-D box. Force law per context: `F = A @ action + B @ velocity`, plus an optional drift. Dynamics: acceleration is 1/32 of the action per step, speed cap 0.1, positions in [-1, 1], 30-step episodes by default, goal radius 0.1, command noise on the action. Three context families, four levels each (12 table contexts): rotations of the commanded force (±30°, ±60°), curl fields (a force perpendicular to velocity, ±7, ±13), and drift (a constant velocity increment in one of four directions). Contexts are never cued; they switch in blocks and must be inferred from errors.

### 1.3 Experiments already in the repository

**Part 1, representation (supervised).** What `z` contains after supervised training on the 12 contexts: context decodability, per-family geometry, the finding that within a family `z` forms a continuous ring (rotations as a ring of directions), and comparisons of the main arm (EC denied the state) against the control arm.

**Part 2, structure learning at the circuit level (`structure.py`).** Train the circuit on a few anchor values of a continuous parameter and test across a continuum; a linear decoder fitted on the anchors predicts held-out values from `z`.

**Part 3, savings and the COIN paradigms (RL on the frozen circuit).** A SAC agent on the frozen circuit, exposed to contexts in blocks. Measures: savings (faster relearning on re-exposure) and COIN's classic paradigms reproduced as block schedules: spontaneous recovery, evoked recovery, savings after washout, with an error-clamp "channel" readout of the expressed memory. Configurations compared: `baseline` (no circuit), `main` and `control` (frozen circuit, two arms), `oracle` (told the context), `coin` (a finite-hypothesis Bayesian observer over the trained contexts, in place of the circuit: the natural COIN baseline, and the generous version of COIN since a Dirichlet-process COIN would spawn a fresh context and start from its prior), plus end-to-end variants.

**Results the author reports from work outside this repository** (described to me, not verified here): the same circuit reproduces the cerebellar dependence of object location memory versus object recognition memory in rodents (a dissociation the model predicted before comparison to the published data), and reproduces the abstract representational geometry reported in hippocampus and prefrontal cortex during serial reversal (Bernardi et al., 2020, Cell).

### 1.4 The new experiment: zero-shot generalization and structure learning (built 2026-09-21/22)

**Design.** The `zero_shot` schedule: the agent trains on four rotations, the WIDE anchor set ±30° and ±90° (chosen so that the midpoints at ±60° are 30° from either anchor; a narrower ±30/±60 set was tried first and dropped because a 50/50 mixture of two neighbouring policies is nearly the right policy when anchors are 30° apart). After training, a FROZEN SWEEP: 16 deterministic centre-out reaches at each of 37 angles (every 5° from −90 to +90), policy frozen, nothing written to long-term memory, block order permuted per seed, under a shortened 12-step horizon (a centre-out reach needs 8 steps at full push; trained policies take 10). Then one FEW-SHOT block: 150 reaches of ordinary training at +60°. The circuit is held out too: the supervised checkpoint was fitted on the four wide rotations plus the curl and drift contexts, so ±15, ±45, ±60, ±75 are novel to the representation as well as to the policy.

**Configurations.** `baseline`; `coin`; `main` on the rotation-trained circuit ("main [trainrotw]"); and the structure-learning control, `main` on a circuit pretrained on four RANDOM LINEAR action transforms at the same effect sizes as the wide rotations ("main [trainlin]", Braun et al. 2009's structured-versus-unstructured contrast). COIN has no circuit, so the pretraining family cannot reach it by construction.

**The figure (figures/sac/zero_shot_h12.pdf; six panels, two rows of three; one shared legend).**
- a. Decoded angle vs true angle: the angle the context code "reads as" (COIN: posterior-mean angle over its hypotheses; circuits: a ridge decoder from reach-mean `z` fitted on training reaches). COIN is a staircase with treads on the anchors; both circuits lie near the diagonal, the unstructured one compressed.
- b. Zero-shot aim: the first action's angle vs rotation, the feedforward compensation carried into a reach before any feedback. COIN aims in steps; the rotation circuit in proportion (about 0.7 of full compensation); baseline aims at −69° everywhere (tuned to the last trained block).
- c. Per-reach distribution of compensation at the ±60° midpoints (kernel density): COIN is bimodal with peaks on the two anchors' compensation and a trough at 60; the rotation circuit is unimodal on 60; the unstructured circuit is a broad hump near 25.
- d. Mode of the per-reach aim vs signed offset from the nearest trained angle, both axes signed: COIN runs along the identity (its mode sits on the nearest anchor, so error = offset); the rotation circuit stays near zero; bootstrap bands.
- e. Share of reaches within 15° of full compensation vs the same offset: COIN is an inverted V (0.8 at the anchors, 0.2 at the midpoints); the rotation circuit flat at about 0.45; the unstructured circuit flat at about 0.28; 95% bands across seeds.
- f. Few-shot learning at +60°: the rotation circuit at ceiling from the first reach (~5.3 return); COIN starts near 4.6 and reaches ceiling over ~150 reaches; baseline climbs from 3.3; the unstructured circuit trails throughout.

Companion figures: circuit_axis_check.pdf (supervised-level gate: decoded angle across the 60° gaps and identification speed within a reach, for both circuits; the rotation circuit reaches a 13° error floor in ~3 steps, the unstructured one 22° in ~5), and standalone versions of panels c, d, e.

**Key quantitative findings.**
- COIN and the rotation circuit are MATCHED on outcomes (success, return, path length) at every angle and horizon. A first-step misaim moves the hand 5% of the reach and feedback absorbs it; no reward re-weighting can make it costly (analysed; only a coarser control rate would). The contrast between them is a contrast of mechanism: categorical choice per trial (COIN) versus continuous position (circuit), visible in the code (a), the feedforward aim (b, d), the per-trial distribution (c, e), and learning speed (f).
- Structure learning is useful, not just present: the unstructured circuit succeeds on 64% of anchor reaches against 92% under the 12-step horizon, return 3.0 vs 5.0, and learns the novel angle slowly where the rotation circuit is at ceiling immediately. Family × model interaction with COIN flat by construction. Caveat: the unstructured circuit is worse at the trained anchors too, so the effect is "family exposure yields a one-parameter code the policy reads cleanly", with generalization as the consequence.
- Precision limitation, to be stated: the circuit's feedforward aim is unbiased and continuous in the angle but its trial-to-trial spread at a trained angle is ~27° against COIN's ~10°, and within one seed and target it is ~20° against 0. The spread is the policy's readout of a carried context state whose decoded angle has ~11° noise; it varies by seed (6° to 31°). Carrying a mean hidden state (Rao-Blackwellising the carry) made it worse and was rejected.

### 1.5 Relevant literature named so far
COIN: Heald, Lengyel & Wolpert (2021, Nature). Structural learning: Braun, Aertsen, Wolpert & Mehring (2009, Current Biology); Braun, Mehring & Wolpert (2010, Behav Brain Res); Turnham, Braun & Wolpert (2011, PLoS Comput Biol). Categorical motor memories and curriculum: Cesanek et al. (2021, eLife) and the 2023 follow-up. Geometry: Bernardi et al. (2020, Cell). Cerebellum and hippocampal spatial code: Rochefort et al. (2011, Science).

## 2. What the author wants to do

1. **Plan the manuscript's structure and high-level figure designs.** The candidate unifying claim: a cerebellar prediction-error signal trains a medial temporal representation that factorizes context from state; cross-condition generalization (Bernardi's geometry), interpolation to untrained rotations (zero-shot axis), and COIN-style contextual inference are that one property seen three ways, and the cerebellar dependence of location memory is what happens when the teaching signal is removed. The author wants help deciding the order of results, what each main figure should show at a glance, and what goes to supplement.
2. **Target venue.** Discussed: Neuron (theory paper) as first target if the memory-dissociation prediction's priority is documented, Nature Neuroscience as alternative, Nature Communications and eLife as fallbacks. The memory data are published data the model predicted, not new experiments. The author has not decided.
3. **Framing decisions already made.** Do not claim "COIN cannot interpolate" or "the circuit beats COIN"; frame COIN as an ideal observer given the plant and the anchors, the circuit as having learned that structure, and the difference as categorical-per-trial versus continuous. Emphasize the structured-versus-unstructured pretraining contrast as the structure-learning result. State the readout-precision limitation. Possibly add a continuous-parameter Bayesian observer as the "structure given, continuous" ceiling to complete a 2×2 (given/learned × discrete/continuous); not yet built.
4. **In progress / planned.** A 50-seed version of the zero-shot sweep is queued (commands ready). Ideas parked: action repeat to make feedforward aim matter in outcomes; a human experiment probing near-anchor angles and trial-by-trial bimodality of initial aim after training on two rotations 60° apart; a control circuit with no action transforms at all.

## 3. Where things are (for reference)
Repository: task.py (plant and contexts), model.py (EC/CB, training, checkpoint conventions), sac.py (agent, configs, schedules incl. `zero_shot`, probes), sweep_ablations.py (run sweeps), zero_shot_analysis.py (the zero-shot figure and companions), circuit_axis_check.py (supervised-level gate), make_ablations_plot.py (savings figures), README.md (documentation of all parts). Figures in figures/sac/. Runs in sac_runs/ (retired variants in subfolders).
