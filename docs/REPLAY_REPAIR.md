# Native demonstration replay

[English](REPLAY_REPAIR.md) | [中文](REPLAY_REPAIR.zh-CN.md)

Dataset v1.1 passes saved-action replay and recorded-state verification for all
2,100 demonstrations in the reference environment. The previous snapshot remains
available at revision `9f48e1404168227a22586af0cffb8371dbb7e1f1`.

## Reference environment

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

## Model corrections

XHand photograph and iPad demonstrations require the corrected left-hand mount.
The model bindings now check its actual orientation as well as revision metadata.
Both groups passed 50/50 unchanged-action replays in the independent reference
environment. The XHand microwave model already uses its corrected mount.

The update contains 220 newly collected physical action sequences. Another 918
sequences retain the original command prefix and remove the unexecuted suffix
after task termination. All 2,100 demonstrations have synchronized observations
exported in the reference environment; the dataset contains 939,356 frames.
These checks certify v1.1 and do not retroactively certify the earlier release.
