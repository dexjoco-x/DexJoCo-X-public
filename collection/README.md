# Scene expansion

**English** | [中文](README.zh.md)

Execute tasks from existing robot demonstrations in randomized scenes, verify success, and export new demonstrations.

Workflow: **Read demonstration → Sample scene → Execute actions → Check success → Export data**.

## Components

| Module | Purpose |
|---|---|
| `dexjoco.benchmark6.environments.make_env` | Create a task and hand environment |
| `dexjoco.autocollect` | Scene expansion and execution |
| `dexjoco.benchmark6.success_policy` | Task success checks |
| `dexjoco.benchmark6.export.export_episode` | Export images, states, actions, and masks |

Check environment connectivity:

```bash
python collection/smoke.py
```

This creates an environment and executes one step. Glove mappings, calibration, and live recording tools are managed separately; this module starts from existing robot demonstrations.

This smoke test checks one simulator step, not end-to-end collection for all 42 combinations. Experimental live-observation adapters are not all implemented. The legacy export_episode entry requires replay.zarr, privileged.zarr and final_truth.json from a collection source; a public training native.npz alone does not provide those inputs.
