"""Prepare FAAS/DexLatent 200-D LeRobot datasets from benchmark7 v8 raw tensors.

Both arms use identical 40-slot blocks: TCP 6D in [base:base+6] and the
shared 32-D hand representation in [base+6:base+38], with bases 0 and 40.
Statistics are fitted on the frozen train split only and videos are hardlinked.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from data_io import link_or_copy


SIDES = ("right", "left")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def statistics(values, masks):
    x = np.concatenate(values)
    m = np.concatenate(masks).astype(bool)
    d = x.shape[-1]
    out = {key: np.zeros(d) for key in ("mean", "std", "min", "max", "q01", "q99", "count")}
    for i in range(d):
        v = x[:, i][m[:, i]]
        if len(v):
            out["mean"][i] = v.mean()
            out["std"][i] = v.std()
            out["min"][i] = v.min()
            out["max"][i] = v.max()
            out["q01"][i], out["q99"][i] = np.quantile(v, (0.01, 0.99))
            out["count"][i] = len(v)
        else:
            out["q01"][i], out["q99"][i], out["std"][i] = -1, 1, 1
    return {key: value.tolist() for key, value in out.items()}


def normalize(x, mask, stats):
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    span = hi - lo
    y = 2 * (x - lo) / np.where(span != 0, span, 1) - 1
    y = np.where(span != 0, y, x)
    return np.where(mask, np.clip(y, -1, 1), 0).astype(np.float32)


class Encoder:
    def __init__(self, representation, contracts, checkpoint=None, seal=None):
        self.representation = representation
        self.contracts = read(contracts)["hands"]
        self.codec_hash = sha(contracts)
        sys.path.insert(0, str(Path(contracts).resolve().parent.parent))
        if representation == "faas":
            from faas7 import FAAS7
            self.codecs = {hand: {side: FAAS7(entry["sides"][side]) for side in SIDES}
                           for hand, entry in self.contracts.items()}
            self.checkpoint_hash = None
        elif representation == "dexlatent":
            import torch
            torch.set_num_threads(int(os.environ.get("DEXLATENT_PREP_THREADS", "32")))
            from dexlatent7 import SidedDexLatent7, HandHead
            proof = read(seal)
            if proof["status"] != "frozen" or proof["test_split_used"]:
                raise ValueError("DexLatent seal is not eligible")
            if sha(checkpoint) != proof["selected_sha256"]:
                raise ValueError("DexLatent checkpoint/seal mismatch")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            canonical = {h: self.contracts[h]["sides"][self.contracts[h]["canonical_side"]]
                         for h in self.contracts}
            if payload.get("contracts") != canonical:
                raise ValueError("DexLatent checkpoint does not match canonical right contracts")
            heads = torch.nn.ModuleDict({h: HandHead(len(canonical[h]["joint_names"]))
                                         for h in self.contracts})
            heads.load_state_dict(payload["heads"])
            self.codecs = {h: {side: SidedDexLatent7(canonical[h], self.contracts[h]["sides"][side], heads[h])
                               for side in SIDES} for h in self.contracts}
            self.checkpoint_hash = sha(checkpoint)
        else:
            raise ValueError(representation)

    def encode(self, raw, raw_mask, key, hand):
        raw = np.asarray(raw)
        raw_mask = np.asarray(raw_mask, dtype=bool)
        y = np.zeros((*raw.shape[:-2], 200), np.float32)
        mask = np.zeros(y.shape, bool)
        for arm, (side, base) in enumerate(zip(SIDES, (0, 40))):
            codec = self.codecs[hand][side]
            contract = self.contracts[hand]["sides"][side]
            active = raw_mask[..., arm, 0]
            y[..., base:base + 3] = raw[..., arm, :3]
            mask[..., base:base + 6] = active[..., None]
            if key == "state":
                quat = raw[..., arm, 3:7].copy()
                quat = np.where(active[..., None], quat, np.array([1, 0, 0, 0]))
                rot = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_rotvec()
                y[..., base + 3:base + 6] = rot.reshape(*raw.shape[:-2], 3)
                n = len(contract["joint_names"])
                hand_value = codec.state(raw[..., arm, 7:7 + n])
            else:
                y[..., base + 3:base + 6] = raw[..., arm, 3:6]
                n = len(contract["actuator_names"])
                hand_value = codec.action(raw[..., arm, 6:6 + n])
            y[..., base + 6:base + 38] = hand_value
            latent_mask = codec.mask if self.representation == "faas" else np.ones(32, bool)
            mask[..., base + 6:base + 38] = active[..., None] & latent_mask
        return np.where(mask, y, 0), mask

    def action_roundtrip(self, raw_action, raw_mask, hand):
        errors = []
        for arm, side in enumerate(SIDES):
            codec = self.codecs[hand][side]
            n = len(self.contracts[hand]["sides"][side]["actuator_names"])
            active = raw_mask[:, arm, 0]
            if not active.any():
                continue
            u = raw_action[active, arm, 6:6 + n]
            decoded = codec.decode(codec.action(u))
            errors.append(np.abs(decoded - u).reshape(-1))
        return np.concatenate(errors) if errors else np.zeros(0)


def main(args):
    root = args.root.resolve()
    bundle = root / "bundle"
    prepared = root / "prepared"
    output = prepared / ("beingh05_" + args.representation)
    if output.exists():
        raise FileExistsError(output)
    encoder = Encoder(args.representation, args.contracts, args.checkpoint, args.seal)
    rows = []
    for path in sorted((bundle / "manifests").glob("*.json")):
        rows.extend(read(path))
    cache = {}
    for row in rows:
        key = row["tensor"]
        if key not in cache:
            with np.load(bundle / key, allow_pickle=False) as z:
                cache[key] = {name: z[name] for name in z.files}

    # Batch by embodiment so the frozen MLP codec runs in a few large calls,
    # rather than thousands of tiny per-episode calls that oversubscribe CPUs.
    encoded = {}
    roundtrip = {}
    hand_rows = {}
    for row in rows:
        hand_rows.setdefault(row["hand"], []).append(row)
    for hand, selected in hand_rows.items():
        lengths = [row["steps"] for row in selected]
        cuts = np.cumsum(lengths)[:-1]
        for name in ("state", "action"):
            raw = np.concatenate([cache[row["tensor"]][name] for row in selected])
            raw_mask = np.concatenate([cache[row["tensor"]][name + "_mask"] for row in selected])
            value, mask = encoder.encode(raw, raw_mask, name, hand)
            for row, part, part_mask in zip(selected, np.split(value, cuts), np.split(mask, cuts)):
                encoded[(row["tensor"], name)] = (part, part_mask)
        raw_action = np.concatenate([cache[row["tensor"]]["action"] for row in selected])
        raw_action_mask = np.concatenate([cache[row["tensor"]]["action_mask"] for row in selected])
        error = encoder.action_roundtrip(raw_action, raw_action_mask, hand)
        if len(error):
            roundtrip[hand] = [error]
        print("ENCODED", args.representation, hand, sum(lengths), flush=True)

    raw_values = {}
    for row in rows:
        if row["split"] == "train":
            group = raw_values.setdefault(row["task"] + "/" + row["hand"],
                {"state": [], "state_mask": [], "action": [], "action_mask": []})
            for name in ("state", "action"):
                value, mask = encoded[(row["tensor"], name)]
                group[name].append(value)
                group[name + "_mask"].append(mask)
    # FAAS and DexLatent are shared semantic spaces.  Per-task/per-hand
    # normalization would remap the same latent value differently in every
    # dataset and quietly undo the intended cross-embodiment alignment.
    shared_stats = {
        name: statistics(
            [chunk for data in raw_values.values() for chunk in data[name]],
            [chunk for data in raw_values.values() for chunk in data[name + "_mask"]],
        )
        for name in ("state", "action")
    }
    stats_name = "beingh05_" + args.representation + "_train_stats.json"
    write(prepared / stats_name, {
        "representation": args.representation,
        "scope": "shared across all task/hand groups; frozen train split only; mask-aware frame pooling",
        "codec_contract_sha256": encoder.codec_hash,
        "codec_checkpoint_sha256": encoder.checkpoint_hash,
        "statistics": shared_stats,
    })
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["hand"], row["split"]), []).append(row)
    manifest = []
    for (task, hand, split), group_rows in sorted(groups.items()):
        dest = output / split / task / hand
        meta = dest / "meta"
        meta.mkdir(parents=True)
        total = 0
        episodes = []
        contract = read(bundle / group_rows[0]["contract"])
        prompt = f"Embodiment: {hand}. {contract['instruction']}"
        views = list(group_rows[0]["videos"])
        norm = shared_stats
        for idx, row in enumerate(group_rows):
            z = cache[row["tensor"]]
            arrays = {}
            for name in ("state", "action"):
                value, mask = encoded[(row["tensor"], name)]
                arrays[name] = normalize(value, mask, norm[name])
                arrays[name + "_mask"] = mask
            n = row["steps"]
            df = pd.DataFrame({
                "observation.state": list(arrays["state"]),
                "action": list(arrays["action"]),
                "observation.state_mask": list(arrays["state_mask"]),
                "action.valid_mask": list(arrays["action_mask"]),
                "timestamp": np.arange(n, dtype=np.float32) / 20,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": idx,
                "index": np.arange(total, total + n, dtype=np.int64),
                "task_index": 0,
            })
            parquet = dest / "data/chunk-000" / f"episode_{idx:06d}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet, index=False)
            episode = {"episode_index": idx, "tasks": [prompt], "length": n,
                "dataset_from_index": total, "dataset_to_index": total + n,
                "source_scene": row["scene_id"], "source_seed": row["seed"],
                "source_split": split, "data/chunk_index": 0, "data/file_index": idx}
            for view in views:
                key = "observation.images." + view
                video = dest / "videos" / key / "chunk-000" / f"episode_{idx:06d}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(bundle / row["videos"][view]["path"], video)
                episode.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": idx,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": (n - 1) / 20})
            episodes.append(episode)
            total += n
        features = {k: {"dtype": "float32", "shape": [200]} for k in ("observation.state", "action")}
        features.update({k: {"dtype": "bool", "shape": [200]} for k in ("observation.state_mask", "action.valid_mask")})
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
        for view in views:
            features["observation.images." + view] = {"dtype": "video", "shape": [256, 256, 3],
                "names": ["height", "width", "channel"], "info": {"video.fps": 20,
                "video.height": 256, "video.width": 256, "video.channels": 3,
                "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "has_audio": False}}
        write(meta / "info.json", {"codebase_version": "v3.0", "robot_type": hand,
            "total_episodes": len(group_rows), "total_frames": total, "total_tasks": 1,
            "chunks_size": 1000, "fps": 20,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
            "features": features, "splits": {split: f"0:{len(group_rows)}"},
            "representation": "benchmark7_" + args.representation + "200_dualarm_v1",
            "codec_contract_sha256": encoder.codec_hash,
            "codec_checkpoint_sha256": encoder.checkpoint_hash,
            "normalization_scope": "shared_train_frame_pool",
            "pre_normalized": True, "object_pose_input": False,
            "layout": "right TCP6+hand32 at 0:38; left TCP6+hand32 at 40:78"})
        identity = {key: [value] * 200 for key, value in (("mean", 0.), ("std", 1.),
            ("min", -1.), ("max", 1.), ("q01", -1.), ("q99", 1.))}
        write(meta / "stats.json", {"observation.state": identity, "action": identity,
            "action.valid_mask": identity})
        write(meta / "raw_train_statistics.json", norm)
        pd.DataFrame({"task_index": [0]}, index=pd.Index([prompt])).to_parquet(meta / "tasks.parquet")
        (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
        episode_dir = meta / "episodes/chunk-000"
        episode_dir.mkdir(parents=True)
        pd.DataFrame(episodes).to_parquet(episode_dir / "file-000.parquet", index=False)
        (meta / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episodes))
        manifest.append({"task": task, "hand": hand, "split": split,
            "path": str(dest.relative_to(root)), "episodes": len(group_rows), "frames": total,
            "config": "benchmark7_native200_dual" if len(views) == 3 else "benchmark7_native200_single"})
        print("PREPARED", args.representation, task, hand, split, len(group_rows), flush=True)
    manifest_name = "beingh05_" + args.representation + "_datasets.json"
    write(prepared / manifest_name, manifest)
    recipe = {}
    for group in manifest:
        if group["split"] != "train":
            continue
        key = "benchmark7_" + group["task"] + "_" + group["hand"]
        recipe[key] = {"dataset_names": [group["hand"]], "data_config_names": [group["config"]],
            "embodiment_tags": ["new_embodiment"], "sampling_strategy": "step",
            "video_backend": "decord", "vit_transform_args": {"transform_type": "default",
            "normalize_type": "imagenet", "use_color_jitter": True}, "frame_step_size": [1],
            "num_used_episodes_per_task": [-1], "num_used_episodes_per_dataset": [-1],
            "num_used_frames_per_dataset": [-1], "weight": group["frames"]}
    write(prepared / ("beingh05_" + args.representation + "_train.yaml"), recipe)
    report = {}
    for hand, chunks in roundtrip.items():
        error = np.concatenate(chunks)
        report[hand] = {"count": len(error), "mae": float(error.mean()),
            "max_abs": float(error.max()), "p99_abs": float(np.quantile(error, 0.99))}
    if args.representation == "faas" and max(x["max_abs"] for x in report.values()) > 1e-5:
        raise ValueError("FAAS roundtrip gate failed")
    write(prepared / ("beingh05_" + args.representation + "_conversion_check.json"), {
        "passed": True, "episodes": len(rows), "train_episodes": sum(r["split"] == "train" for r in rows),
        "two_arm_encoding": True, "test_split_used_for_statistics": False,
        "roundtrip": report, "manifest": manifest_name})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--representation", choices=("faas", "dexlatent"), required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seal", type=Path)
    main(parser.parse_args())
