"""Read physical Native fields and RGB from one downloaded demonstration."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset.tools.adapters import encode


def read_demo(data_root, task, hand, scene="scene_000", frame=0, horizon=25):
    """Return unnormalized arm40 fields; padded future actions have false masks."""
    if frame < 0 or horizon < 1:
        raise ValueError("frame must be nonnegative and horizon must be positive")
    bundle = Path(data_root) / "bundle"
    manifest = bundle / "manifests" / (task + ".json")
    rows = json.loads(manifest.read_text())
    matches = [row for row in rows if row["hand"] == hand and row["scene_id"] == scene]
    if len(matches) != 1:
        raise ValueError("Select one installed hand/scene from the task manifest")
    row = matches[0]
    if frame >= row["steps"]:
        raise ValueError("frame is outside the demonstration")
    contract = json.loads((bundle / row["contract"]).read_text())
    count = min(horizon, row["steps"] - frame)
    with np.load(bundle / row["tensor"], allow_pickle=False) as raw:
        state, state_mask = encode(raw["state"][frame], raw["state_mask"][frame],
                                   "state", "beingh05")
        actions, action_mask = encode(raw["action"][frame:frame + count],
                                     raw["action_mask"][frame:frame + count],
                                     "action", "beingh05")
        timestamp = float(raw["timestamp"][frame])
    # The converter's physical fields occupy the first 80 of its 200 slots.
    state, state_mask = state[:80].copy(), state_mask[:80].copy()
    actions = np.pad(actions[:, :80], ((0, horizon - count), (0, 0)))
    action_mask = np.pad(action_mask[:, :80], ((0, horizon - count), (0, 0)))
    images = {}
    for view, video in row["videos"].items():
        capture = cv2.VideoCapture(str(bundle / video["path"]))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError("Cannot decode video frame: " + view)
            images[view] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
    return {
        "state": state, "state_mask": state_mask,
        "actions": actions, "action_mask": action_mask,
        "images": images, "instruction": contract["instruction"],
        "timestamp": timestamp, "valid_action_steps": count,
        "episode_steps": row["steps"], "control_dt": contract["control_dt"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--task", default="hammer_nail")
    parser.add_argument("--hand", default="allegro")
    parser.add_argument("--scene", default="scene_000")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--output", type=Path, help="Save RGB previews to a new directory")
    args = parser.parse_args()
    sample = read_demo(args.data_root, args.task, args.hand, args.scene,
                       args.frame, args.horizon)
    summary = {
        "task": args.task, "hand": args.hand, "scene": args.scene,
        "frame": args.frame, "episode_steps": sample["episode_steps"],
        "instruction": sample["instruction"], "layout": "arm40_v1",
        "normalized": False, "side_order": ["right", "left"],
        "state_shape": list(sample["state"].shape),
        "actions_shape": list(sample["actions"].shape),
        "active_state_coordinates": int(sample["state_mask"].sum()),
        "active_action_coordinates": int(sample["action_mask"][0].sum()),
        "valid_action_steps": sample["valid_action_steps"],
        "timestamp": sample["timestamp"], "control_dt": sample["control_dt"],
        "rgb": {view: list(image.shape) for view, image in sample["images"].items()},
    }
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for view, image in sample["images"].items():
            if not cv2.imwrite(str(args.output / (view + ".png")),
                               cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError("Could not save RGB preview: " + view)
        (args.output / "sample.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
