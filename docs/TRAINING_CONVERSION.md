# Training action conversion

[English](TRAINING_CONVERSION.md) | [中文](TRAINING_CONVERSION.zh.md)

## Download verified Training v1

Training v1 contains **2,100 re-recorded trajectories and 933,865 frames**, with
50 demonstrations for every hand–task pair. All final saved-label sequences pass
continuous task-success and recorded-state checks in the reference environment.
Full Being-H and official LeRobot 0.4.4 reader checks also pass.

The package is 5.21 GB and includes synchronized videos, raw tensors, frozen
statistics, and both prepared readers. Training v1 is the current benchmark
standard for replay and new training. Download into a new directory; earlier
snapshots are archived only for tracing prior results.

```bash
export DATA_ROOT="$PWD/outputs/training_v1"
python tools/dataset_packages.py download --tasks all --data-root "$DATA_ROOT"
```

Repeated videos use TAR hard links; extraction produces ordinary local files,
with no symbolic links. The prepared views are `prepared/beingh_final_v3/` and
`prepared/lerobot_final_v3/`. Their corresponding statistics are in
`prepared/beingh_final_v3_train_stats.json` and
`prepared/lerobot_final_v3_train_stats.json`.

In the reference physics environment:

```bash
python -m pip install --no-deps -r requirements/training-replay-pip.txt
python evaluation/audit_training_replay.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_final_v3 --workers 4 --output training_replay.json
```

The package's `metadata/VERIFICATION.json` and per-episode report record the
checks. These checks execute demonstration labels, not a trained model. No task
success criterion or runtime model was changed for the training repair.

For official LeRobot, use a separate reader/training environment:

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

folder = Path("outputs/training_v1/prepared/lerobot_final_v3/train/hammer_nail/allegro")
windows = {key: [i / 20 for i in range(16)]
           for key in ("action", "action.valid_mask")}
dataset = LeRobotDataset("dexjoco-x/local", root=folder,
                        video_backend="pyav", delta_timestamps=windows)
sample = dataset[0]
loss_mask = sample["action.valid_mask"] & ~sample["action_is_pad"][:, None]
```

The native200 labels are already normalized. Disable further state/action
normalization in the training configuration. Use `Native200PolicyAdapter` below
with the matching statistics, and retain its float64 commands through the
simulation interface. Existing checkpoints keep their original decoder and
statistics. This verified package contains the native200 representation.

The old conversion clipped valid demonstrations at dataset quantiles. For example,
a 1.61 cm wrist command could become 0.41 cm. The new `quantile_affine_v3` scales
commands without removing values outside the central 98% of the data. Labels can
exceed [-1, 1]. The policy's physical controller limits still apply.

V3 also scales labels by powers of two so nearly constant channels do not produce
huge training targets. The decoder reverses this scaling before the affine inverse.
For collected v2 labels, `rescale_training_bundle.py` checks every decoded command
for exact equality and retains the matching videos and states. Its scale fitting
also preserves tiny commands that would otherwise underflow during rescaling.

Statistics record the normalization version, and new outputs use separate paths.
Unversioned statistics retain `quantile_clip_v1` behavior for old checkpoints.
Do not switch a trained checkpoint to different statistics or a different decoder.

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output data_check.json
```

The output is `prepared/beingh05_affine_v3/`; its statistics, dataset list and
training recipe have the same prefix. Use `--output-name NAME` for another export.
For Training v1, the converter automatically loads `bundle/training_statistics.json`.
Raw conversion experiments require `--allow-unverified-conversion`; their labels
still need independent replay and matching observations before use as a verified
training release.
Use `--statistics-file FILE` to reuse frozen statistics when exporting newly
collected trajectories. Fit statistics once, before collecting final labels.

The new decoder uses float64 arithmetic after reading float32 labels. Equal
quantiles use `label = value - q01`, so rare motion in otherwise constant channels
is retained. At actual actuator limits, labels round inward to avoid an extra
clamp changing the decoded command. This is distinct from dataset-quantile clipping.

These numerical checks do not establish physical success. In the
[reference environment](REPLAY_REPAIR.md), install the additional Parquet reader
and replay the exported files themselves:

```bash
python -m pip install --no-deps -r requirements/training-replay-pip.txt
python evaluation/audit_training_replay.py --data-root "$DATA_ROOT" \
  --prepared-name beingh05_affine_v3 --workers 4 --output training_replay.json
```

The audit separately reports task completion and agreement with recorded states.
If converted commands produce a different trajectory, regenerate its RGB and
state observations from the actual execution. A newly collected episode must
execute the decoded final labels, save those labels, and pass an independent
replay from the saved training files. Conversion alone does not certify Raw v1.1
as a float32 training release.

The two readers use different path templates. The default export retains the
episode paths required by Being-H. Use `--reader lerobot` for official LeRobot v3
file-index paths. Both exports retain the same labels and share the video files.
In a separate reader environment with `lerobot==0.4.4`:

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT" --reader lerobot \
  --statistics-file "$DATA_ROOT/prepared/beingh05_affine_v3_train_stats.json"
python tools/check_lerobot.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_affine_v3 --output lerobot_reader_check.json
```

This checks every numeric label, sampled RGB frames, and action chunks at episode
boundaries. LeRobot repeats terminal values for padded action steps; exclude them
using `action_is_pad`, together with `action.valid_mask`. Being-H's adapter instead
zeros its padded steps. The stored labels are already normalized; use their bound
decoder and do not normalize them again.

For a newly trained native200 policy, use `Native200PolicyAdapter` at an
`arm40_v1` policy server boundary:

```python
from dataset.tools.policy_adapter import Native200PolicyAdapter

adapter = Native200PolicyAdapter(training_statistics['groups'])
state, state_mask = adapter.state(observation)
# prediction: float32 [horizon, 200], from the model trained on these labels
reply = {'actions': adapter.actions(prediction, observation)}
```

Send the returned float64 commands unchanged. This does not modify the pi0.5
example or existing checkpoint decoders; it is the adapter for the new native200
training representation.

For newly collected episodes, `native.npz` also stores the exact `training_labels`
executed by the collector. Both the exporter and portable loader retain these
labels verbatim and reject mismatched statistics. RGB and robot states must come
from that same rollout. These are new training trajectories, not a claim that
float32 conversion preserves the original float64 execution.

`collection/collect_training_labels.py` can track the accepted source wrist
targets while executing the final labels. For hammer collection, the optional
`--hammer-strokes` adds explicit vertical strokes after the source target sequence.
Those steps are part of the saved demonstration and are included in its new RGB
and state recordings. The collector records the parameters and independently
replays the result; the evaluation runtime never appends recovery actions.

This check executes saved labels as ideal commands; it does not measure a trained
policy's success rate. The dataset, model architecture, and training budget remain
separate experimental variables.
