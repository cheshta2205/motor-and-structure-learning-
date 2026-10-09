# Project overview

## The idea in one paragraph

A cerebellar forward model predicts the sensory consequences of a movement. Its prediction error is the only signal a recurrent medial-temporal-lobe network receives about the present, and from that error alone the network infers which context the body is in: which force field, which visuomotor rotation. The context it infers is handed to a motor controller that learns by reinforcement. The claim is that this one loop, cerebellar prediction error teaching a medial temporal representation, accounts for a set of results that are currently explained separately: contextual inference in human motor adaptation, the cerebellar dependence of certain memory tasks in rodents, the abstract representational geometry recorded in hippocampus and prefrontal cortex, and structural learning, the ability to generalize to a member of a task family never practised.

## The model

Three modules. CB, a cerebellum-like forward model: next state from current state, action and a context latent. EC, an entorhinal-like recurrent network that turns the stream of cerebellar prediction errors into a context latent z; in the main arm it never sees the state directly. MC, a motor cortex stand-in: a soft actor-critic agent that controls a point mass using the state, the goal and z. EC and CB are trained by supervised next-state prediction on sequences of switching contexts and then frozen; MC is trained on top. A control arm gives EC the state as well, to ask whether the prediction error is load-bearing.

## The task

Two-dimensional reaching in a box, with uncued contexts that switch in blocks: rotations of the commanded force, curl fields perpendicular to velocity, and constant drifts, four levels each. The dynamics are slow, acceleration is a thirty-second of the command per step, so feedback within a reach can absorb most of a wrong first push; this matters for interpreting outcome measures.

## What has been shown

1. **Representation.** After supervised training, z decodes context; within a family it forms a continuous ring rather than a set of points, so a rotation is a position on an axis.
2. **Contextual inference behaviour.** With a reinforcement-learning controller on the frozen circuit, the model reproduces savings on re-exposure and COIN's paradigms: spontaneous recovery, evoked recovery, savings after washout, read out through an error-clamp channel. A finite-hypothesis Bayesian observer stands in for COIN as the comparison.
3. **Zero-shot generalization and structure learning.** Trained on rotations at ±30° and ±90°, the circuit's code and its feedforward aim vary continuously with a novel rotation; COIN's are a staircase, committing to the nearest known context on each trial, so that at the ambiguous midpoint its aim is bimodal while the circuit's is unimodal on target. Outcomes are matched, because the plant's feedback absorbs a misaim; the difference is in mechanism and in learning speed at a novel angle. A circuit pretrained on unstructured random transforms at matched effect sizes, the classic structural-learning control, carries the angle only compressed and noisily, compensates half as much, and learns the novel angle slowly. Pretraining on the family is what makes the code a usable axis; COIN cannot benefit from pretraining by construction.
4. **Outside the motor task** (the author's work, not in this repository): the same circuit predicted the cerebellar dependence of object location memory but not object recognition memory in rodents, confirmed against published data, and reproduces the abstract geometry with cross-condition generalization reported in hippocampus and prefrontal cortex during serial reversal.

## The candidate unifying claim

The cerebellar prediction error trains a medial temporal representation that factorizes context from state. Cross-condition generalization in recordings, interpolation to an untrained rotation, and COIN-style contextual inference are that one property seen three ways; the cerebellar dependence of location memory is what happens when the teaching signal is removed.

## Known limitations to state

The circuit's trial-to-trial precision at a trained context is about three times worse than an ideal observer's, a property of how the controller reads a noisy carried context state. Neither the circuit nor COIN extrapolates beyond the outermost trained rotation. The blind baseline's behaviour after training is tuned to whichever context came last.

## Status

Model, tasks and analyses are complete for the motor results; a 50-seed version of the zero-shot sweep is queued. A continuous-parameter observer as a "structure given, continuous" ceiling has been discussed and not built. Venue undecided among Neuron, Nature Neuroscience, Nature Communications and eLife.
