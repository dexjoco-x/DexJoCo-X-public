# Versions and compatibility

[中文](VERSIONING.md) | **English**

| Version | Contents |
|---|---|
| `paper-v1` | Training data, checkpoints, and evaluation results for Table I in the paper |
| `dataset-v1.0-corrected` | The current public dataset of 2,100 demonstrations |
| `runtime-v9` | Simulation environments matching the current dataset |

The public dataset updates 50 XHand microwave demonstrations. The paper checkpoints were trained before this update and have not been retrained on the current public dataset.

`linkerhand_l20` is a compatibility key in files and contracts. Its source left/right URDFs were modified from the official L25 files. Documentation uses `LinkerHand`; this release preserves the key and model geometry.
