#!/usr/bin/env python3
"""
Convert SO-101 / piper HDF5 episodes -> LeRobot **v2.0** dataset for DreamZero.

We do NOT use the `lerobot` library: recent versions emit the v3.0 aggregated
layout (file-000.parquet / tasks.parquet) + AV1 video, which DreamZero's
converter and its decord-based loader cannot read. This writes the exact v2.0
layout DreamZero expects, with H.264 video (decord-friendly):

    out/
      data/chunk-000/episode_000000.parquet ...
      videos/chunk-000/observation.images.cam_global/episode_000000.mp4 ...
      videos/chunk-000/observation.images.cam_gripper/episode_000000.mp4 ...
      meta/info.json
      meta/episodes.jsonl
      meta/tasks.jsonl

Source HDF5 (one file per episode):
    actions/position                 (T, 7)  float32
    observations/follower/position   (T, 7)  float32
    observations/images/<cam>        (T, H, W, 3) uint8
"""
import argparse
import glob
import json
import os
import subprocess

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNKS_SIZE = 1000  # all 407 episodes land in chunk-000


def encode_mp4(frames: np.ndarray, fps: int, out_path: str):
    """frames: (T, H, W, 3) uint8 RGB -> H.264 yuv420p mp4 via ffmpeg stdin."""
    T, H, W, _ = frames.shape
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-g", "1",  # all-intra: every frame seekable (DreamZero samples arbitrary frames)
        out_path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", required=True)
    ap.add_argument("--state-key", default="observations/follower/position")
    ap.add_argument("--action-key", default="actions/position")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    files = sorted(glob.glob(os.path.join(src, "**", "*.h*5"), recursive=True))
    if args.limit:
        files = files[: args.limit]
    assert files, f"No .hdf5/.h5 files under {src}"
    print(f">> {len(files)} episodes -> {out}")

    with h5py.File(files[0], "r") as h:
        sdim = int(h[args.state_key].shape[1])
        adim = int(h[args.action_key].shape[1])
        cams = list(h["observations/images"].keys())
        cam_shapes = {c: tuple(int(x) for x in h[f"observations/images/{c}"].shape[1:]) for c in cams}
    print(f">> state_dim={sdim} action_dim={adim} cameras={cam_shapes}")

    os.makedirs(os.path.join(out, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)

    ep_meta = []
    global_index = 0
    total_frames = 0

    for ep_idx, fp in enumerate(files):
        with h5py.File(fp, "r") as h:
            state = np.asarray(h[args.state_key], dtype=np.float32)
            action = np.asarray(h[args.action_key], dtype=np.float32)
            imgs = {c: np.asarray(h[f"observations/images/{c}"]) for c in cams}
        T = min(len(state), len(action), *[len(imgs[c]) for c in cams])
        state, action = state[:T], action[:T]

        # videos: videos/chunk-000/observation.images.<cam>/episode_XXXXXX.mp4
        for c in cams:
            vid = os.path.join(out, "videos", "chunk-000", f"observation.images.{c}",
                               f"episode_{ep_idx:06d}.mp4")
            encode_mp4(imgs[c][:T], args.fps, vid)

        # parquet row group
        cols = {
            "observation.state": [state[t] for t in range(T)],
            "action": [action[t] for t in range(T)],
            "timestamp": np.arange(T, dtype=np.float32) / args.fps,
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, ep_idx, dtype=np.int64),
            "index": np.arange(global_index, global_index + T, dtype=np.int64),
            "task_index": np.zeros(T, dtype=np.int64),
        }
        table = pa.table({
            "observation.state": pa.array(cols["observation.state"], type=pa.list_(pa.float32())),
            "action": pa.array(cols["action"], type=pa.list_(pa.float32())),
            "timestamp": pa.array(cols["timestamp"]),
            "frame_index": pa.array(cols["frame_index"]),
            "episode_index": pa.array(cols["episode_index"]),
            "index": pa.array(cols["index"]),
            "task_index": pa.array(cols["task_index"]),
        })
        pq.write_table(table, os.path.join(out, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet"))

        ep_meta.append({"episode_index": ep_idx, "tasks": [args.task], "length": T})
        global_index += T
        total_frames += T
        print(f"[{ep_idx + 1}/{len(files)}] {os.path.basename(fp)}  T={T}")

    # ---- meta/info.json (LeRobot v2.0) ----
    features = {
        "observation.state": {"dtype": "float32", "shape": [sdim],
                              "names": [f"motor_{i}" for i in range(sdim)]},
        "action": {"dtype": "float32", "shape": [adim],
                   "names": [f"motor_{i}" for i in range(adim)]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for c in cams:
        H, W, C = cam_shapes[c]
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": [H, W, C],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": float(args.fps), "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.height": H, "video.width": W,
                     "video.channels": C, "has_audio": False},
        }
    info = {
        "codebase_version": "v2.0",
        "robot_type": "so101",
        "total_episodes": len(files),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(files) * len(cams),
        "total_chunks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{len(files)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(os.path.join(out, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as f:
        for e in ep_meta:
            f.write(json.dumps(e) + "\n")

    print(f">> DONE. LeRobot v2.0 dataset at: {out}")


if __name__ == "__main__":
    main()
