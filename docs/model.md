# Model

The checkpoint architecture is `withdrawal_expert_v1`. A recurrent base policy and a withdrawal decoder share the causal observation history through the model implementation; a learned router selects the action branch. A separate learned head predicts phase. This is a state-input action-chunking policy adapted for this robot, not a direct reproduction of visual ACT.

The 64 inputs comprise the current 17 joint states, the previous 17 command values, a 24-dimensional target-geometry descriptor and six palm-relative geometry features. The earlier 13-dimensional external phase one-hot is not supplied to this model. An XYZ coordinate padded with zeros is not a valid target descriptor.

The model predicts 24 steps of 17 joint values. Calibrated curl/release branches map the three coupled joints of each finger to one actuator. `core.py` constructs features, loads the network, updates causal memory and projects commands. `frozen/` contains the network and geometry helpers, `weights/` contains the checkpoint and normalization, and `assets/calibration/` contains the coupled mapping.

Model Home is `[0, 55, -156, 0, 0]` degrees. These are model joint angles, not uncalibrated servo or PWM commands.

The final ACT training used all 1,315 verified records. The historical split and subsequent all-training use must be distinguished. Numerical replay of an included observation sequence is not held-out generalization evaluation. No residual reinforcement-learning policy is included in this checkpoint.

Action generation does not guarantee two completed rounds, adequate motion of every finger, desired surface distance or a correct Home return at every target. Report those outcomes separately from the model's terminal prediction.
