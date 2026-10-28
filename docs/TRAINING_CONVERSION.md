# Training action conversion

[English](TRAINING_CONVERSION.md) | [中文](TRAINING_CONVERSION.zh.md)

The old conversion clipped valid demonstrations at dataset quantiles. For example,
a 1.61 cm wrist command could become 0.41 cm. The new `quantile_affine_v2` scales
commands without removing values outside the central 98% of the data. Labels can
exceed [-1, 1]. The policy's physical controller limits still apply.

Statistics record the normalization version, and new outputs use separate paths.
Unversioned statistics retain `quantile_clip_v1` behavior for old checkpoints.
Do not switch a trained checkpoint to different statistics or a different decoder.

```bash
python dataset/tools/prepare_server.py "$DATA_ROOT"
python tools/dataset_smoke.py --data-root "$DATA_ROOT" --output data_check.json
```

The output is `prepared/beingh05_affine_v2/`; its statistics, dataset list and
training recipe have the same prefix. Use `--output-name NAME` for another export.
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
  --prepared-name beingh05_affine_v2 --workers 4 --output training_replay.json
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
  --statistics-file "$DATA_ROOT/prepared/beingh05_affine_v2_train_stats.json"
python tools/check_lerobot.py --data-root "$DATA_ROOT" \
  --prepared-name lerobot_affine_v2 --output lerobot_reader_check.json
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

This check executes saved labels as ideal commands; it does not measure a trained
policy's success rate. The dataset, model architecture, and training budget remain
separate experimental variables.
