# Demonstration replay

**English** | [中文](REPLAY_REPAIR.zh-CN.md)

Use **Training v1** for both training and replay. [Download the current dataset](../dataset/README.md).
All 2,100 final saved-label sequences complete their tasks and pass recorded-state
checks in the reference environment. Their decoded commands exactly match the
package's float64 actions, and all readers use identical videos.

## Reference environment

Update the code before replaying Training v1. Earlier controllers could produce
different trajectories on different CPUs even when the package and BLAS checks
passed: NumPy selected different inverse-trigonometric implementations. The
controller now uses scalar rotation math with a fixed evaluation order. Audit
reports also record NumPy CPU dispatch, system library hashes and the controller
math revision. This code correction does not replace the Training v1 archive or
videos, change task success rules, or relax the `2e-5` state tolerance.
The corrected controller passes all 2,100 native replays with NumPy AVX-512
disabled, plus 84 cases covering all hand/task groups on Ubuntu 22.04.
[Validation summary](replay/controller_math_v1.json).

Linux x86_64 with AVX2/FMA and Conda is required. Create a separate environment:

```bash
conda create -n dexjoco-replay --file requirements/native-replay-linux64.conda.txt
conda env config vars set -n dexjoco-replay OPENBLAS_CORETYPE=Haswell \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
conda activate dexjoco-replay
python -m pip install -r requirements/native-replay-pip.txt
python -m pip install --force-reinstall --no-deps numpy==1.26.4 PyOpenGL==3.1.10
```

MuJoCo package version alone does not identify its binary build. This recipe pins
the Conda-forge build and uses the NumPy wheel. The audit launcher fixes the BLAS
CPU kernel and thread count before importing the numerical libraries.

## Audit saved actions

```bash
python tools/replay_reference.py /path/to/dataset --workers 4 --output /tmp/native-replay.json
```

The dataset directory must contain `bundle/manifests/`, or be `bundle/` itself.
The audit checks every manifest entry, including training demonstrations. Use
`--task`, `--hand`, or `--scene` to select a smaller scope. Results are written
per episode while the audit runs; exit status is zero only when every selected
demonstration succeeds.

Add `--check-states` to require every recorded pre-action TCP pose and hand joint
state to match the replay within `2e-5`. FK is computed on separate render data;
this check does not reset or modify the physics state.

The audit verifies tensor checksums, the initial robot state and model bindings,
then executes saved actions continuously until termination, truncation or the
end of the sequence. Success must be observed during that sequence.

## Audit saved LeRobot labels

```bash
python -m pip install --no-deps -r requirements/training-replay-pip.txt
python evaluation/audit_training_replay.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_final_v3 --workers 4 --output training_replay.json
```

The labels are float32; decode them with the package's statistics and execute the
float64 commands. See [training usage](TRAINING_CONVERSION.md). These are
demonstration-replay checks, not trained-policy success rates. Earlier precision
diagnostics and corrections are in [version history](VERSIONING.md); they do not
describe the current Training v1 result.
