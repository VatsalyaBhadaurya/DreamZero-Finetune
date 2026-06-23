#!/usr/bin/env python3
"""
Convert SO-101 / piper HDF5 episodes -> LeRobot v2.0 dataset (for DreamZero).

Source layout (one file per episode), as found in so101_hdf:
    actions/position                 (T, 7)  float32   -> action
    observations/follower/position   (T, 7)  float32   -> observation.state
    observations/images/cam_global   (T, 720, 1280, 3) uint8 -> video
    observations/images/cam_gripper  (T, 480, 640, 3)  uint8 -> video
    (velocity arrays are empty -> ignored)

Usage:
    python scripts/hdf5_to_lerobot.py \
        --src  ~/Desktop/R/piper_datasets/so101_hdf \
        --out  ~/Desktop/R/piper_datasets/so101_lerobot \
        --task "pick the block and drop it in the box" \
        --fps  30
Add --limit 3 first for a quick smoke test before the full 407-episode run.
"""
import argparse
import glob
import inspect
import os

import h5py
import numpy as np


def get_lerobot_dataset_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # newer layout
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # older layout
    return LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing episode_*.hdf5")
    ap.add_argument("--out", required=True, help="output LeRobot dataset root")
    ap.add_argument("--repo-id", default="so101/pick_block")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", required=True, help="language instruction for every episode")
    ap.add_argument("--state-key", default="observations/follower/position")
    ap.add_argument("--action-key", default="actions/position")
    ap.add_argument("--limit", type=int, default=0, help="only convert first N episodes (debug)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.src), "**", "*.h*5"), recursive=True))
    if args.limit:
        files = files[: args.limit]
    assert files, f"No .hdf5/.h5 files under {args.src}"
    print(f">> {len(files)} episodes")

    # Infer dims + cameras from the first file.
    with h5py.File(files[0], "r") as h:
        sdim = int(h[args.state_key].shape[1])
        adim = int(h[args.action_key].shape[1])
        cams = list(h["observations/images"].keys())
        cam_shapes = {c: tuple(int(x) for x in h[f"observations/images/{c}"].shape[1:]) for c in cams}
    print(f">> state_dim={sdim} action_dim={adim} cameras={cam_shapes}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (sdim,),
                              "names": [f"motor_{i}" for i in range(sdim)]},
        "action": {"dtype": "float32", "shape": (adim,),
                   "names": [f"motor_{i}" for i in range(adim)]},
    }
    for c in cams:
        h_, w_, ch_ = cam_shapes[c]
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": (h_, w_, ch_),
            "names": ["height", "width", "channel"],
        }

    LeRobotDataset = get_lerobot_dataset_cls()
    out = os.path.expanduser(args.out)
    try:
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, root=out,
                                   features=features, use_videos=True)
    except TypeError:
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, root=out,
                                   features=features)

    add_has_task = "task" in inspect.signature(ds.add_frame).parameters
    save_has_task = "task" in inspect.signature(ds.save_episode).parameters

    for fi, fp in enumerate(files):
        with h5py.File(fp, "r") as h:
            state = np.asarray(h[args.state_key], dtype=np.float32)
            action = np.asarray(h[args.action_key], dtype=np.float32)
            imgs = {c: np.asarray(h[f"observations/images/{c}"]) for c in cams}
            T = min(len(state), len(action), *[len(imgs[c]) for c in cams])
            for t in range(T):
                frame = {"observation.state": state[t], "action": action[t]}
                for c in cams:
                    frame[f"observation.images.{c}"] = imgs[c][t]
                if add_has_task:
                    ds.add_frame(frame, task=args.task)
                else:
                    frame["task"] = args.task
                    ds.add_frame(frame)
            ds.save_episode(task=args.task) if save_has_task else ds.save_episode()
        print(f"[{fi + 1}/{len(files)}] {os.path.basename(fp)}  T={T}")

    if hasattr(ds, "consolidate"):
        try:
            ds.consolidate()
        except Exception as e:
            print("consolidate skipped:", e)

    print(f">> DONE. LeRobot dataset at: {out}")


if __name__ == "__main__":
    main()
