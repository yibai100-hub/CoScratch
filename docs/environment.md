# Environment and configuration

## Inference and offline checks

- Python 3.11.15
- PyTorch 2.7.0+cu128; the numerical replay also runs on CPU
- NumPy 1.26.0 and SciPy 1.15.3
- Dependencies: `act/requirements-pc.txt`
- Model contract: `act/config/model_contract.json`
- Kinematics: `act/assets/robot_kinematics.urdf`
- Coupled mapping: `act/assets/calibration/motor_to_joint.json`

## Geometric planning and PhysX

The historical local runtime uses Windows and Isaac Sim 5.1. The controller requires the Isaac USD, PhysX, Lula and SingleArticulation modules. It cannot run in ordinary Python. It also imports Windows `msvcrt`; direct Linux compatibility has not been established for this published controller.

The robot scene root is `/World/robotnew_with_aero_hand_right`. The scene must contain the expected joints, collision shapes, back surface and coordinate relationships. A URDF alone is not the complete USD scene. The included candidate-point CSV and model frame configuration refer to the original setup and must be reconciled before changing scenes.

The ACT simulation configuration used 120 Hz control, 240 Hz physics and solver iterations 128/4. Back-surface physical collision was disabled and selected internal hand collisions were excluded. Arm self-collision and physical joint ranges remained enabled. These demonstration settings are not evidence of physical contact validation.

## Outstanding release work

1. Review redistribution terms for the full USD scene, external materials and robot meshes.
2. Run a native PhysX regression of the renamed and relocated controller in the matching scene.
3. If training reproduction is required, publish the dataset index, training stages, seeds and training scripts separately. This package supports inference inspection and replay.
4. Add formal paper metadata, citation details and a confirmed source-code license.

Do not commit a complete Isaac, CUDA or Python installation, cloud accounts or device credentials. Distribute dependency specifications and the required project resources instead.
