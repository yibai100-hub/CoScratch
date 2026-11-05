# ACT inference package

This package contains the final `withdrawal_expert_v1` checkpoint, its normalization statistics and the matching frozen model definitions. The base branch, withdrawal branch, recurrent memory and learned router are stored in one checkpoint; this is not a concatenation of prerecorded trajectories.

## Files

- `core.py`: feature construction, recurrent inference, actuator projection and command-state bookkeeping.
- `frames.py`: explicit camera, robot-base and model coordinate transforms.
- `frozen/`: model definitions, geometry helpers and coupled-finger mappings.
- `assets/`: model URDF, curl/release calibration and initial pose.
- `gateway.py`: a mock communication gateway for local protocol tests.
- `hardware_adapter.py`: an SDK injection interface; it does not enable physical motion by default.
- `tests/` and `fixtures/`: contract tests and recorded simulation observations.
- `reference/`: historical calibration files and simulation evidence, not active hardware calibration.

## Run

```sh
python validate_pc.py
python pc_mock_demo.py
```

Alternatively use `validate-pc.ps1 -Python /path/to/python.exe` on Windows. Results appear in `reports/`. These checks exercise numerical replay and a loopback mock device, not native physics or real hardware.

## Observation and execution contract

Input features contain 17 current joint values, 17 previous command values, 24 target-geometry values and six palm-relative features. No external phase one-hot or clock-based phase switch is provided. The model predicts 24 steps; recurrent observation cadence is 12 control steps.

On the intended physical device, J1/J3/J4/J5 use timestamped position readings. J2 and the four fingers use estimates derived from the last acknowledged command. Each finger has one actuator controlling three coupled joints. Estimates must remain explicitly labeled; queued predictions or a successful ROS publish are not device acknowledgements. Failed writes must not advance the command ledger.

## Calibration boundary

Model Home remains `[0, 55, -156, 0, 0]` degrees. Historical ROS and joint-mapping files contain different offset and Home conventions; they are retained as reference and are not selected automatically. `config/hardware_binding.pending.json` leaves unverified offsets and directions unset. Real execution requires matching the current installation, SDK, actuator IDs, coupled curves and hand-eye calibration.

A mock STOP only stops mock dispatch. Physical emergency stop, thermal handling, feedback latency, exclusive bus ownership and collision behavior require device-level integration. A reference observation replay cannot establish the behavior of a closed loop using estimated J2/finger states.
