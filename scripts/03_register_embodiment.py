#!/usr/bin/env python3
"""
Step 3: Register your embodiment in the DreamZero repo.

Automates the safe parts:
  * adds your tag to groot/vla/data/schema/embodiment_tags.py
  * adds your tag to VALID_EMBODIMENT_TAGS in the converter
  * writes groot/vla/configs/data/dreamzero/<EMB>_relative.yaml
  * emits a ready-to-paste modality+transform block for the base config
    (you paste this once, by hand, because the base YAML uses local anchors).

Run AFTER sourcing config.env so EMB / NUM_VIEWS / DREAMZERO_ROOT are set:
    source scripts/config.env && python scripts/03_register_embodiment.py
"""
import os
import re
import sys
from pathlib import Path

EMB = os.environ["EMB"]
NUM_VIEWS = int(os.environ.get("NUM_VIEWS", "2"))
ROOT = Path(os.environ["DREAMZERO_ROOT"])
FPS = int(os.environ.get("FPS", "30"))

# Sub-keys derived from config.env (names only).
REL_KEYS = os.environ.get("RELATIVE_ACTION_KEYS", "joint_pos gripper_pos").split()
# Camera names must match the LeRobot observation.images.<name> keys (and thus
# the GEAR modality.json video sub-keys). Falls back to cam0..camN if unset.
_cam_names = os.environ.get("CAM_NAMES", "").split() or [f"cam{i}" for i in range(NUM_VIEWS)]
VIDEO_KEYS = [f"video.{c}" for c in _cam_names]
STATE_MKEYS = [f"state.{k}" for k in REL_KEYS]
ACTION_MKEYS = [f"action.{k}" for k in REL_KEYS]


def patch_enum():
    f = ROOT / "groot/vla/data/schema/embodiment_tags.py"
    txt = f.read_text()
    line = f'    {EMB.upper()} = "{EMB}"'
    if line in txt:
        print(f"   enum: {EMB.upper()} already present")
        return
    # insert a new member right after the class declaration line
    m = re.search(r"(class\s+EmbodimentTag\b[^\n]*\n)", txt)
    if not m:
        print("   !! could not find EmbodimentTag class; add manually:", line)
        return
    txt = txt[: m.end()] + line + "\n" + txt[m.end():]
    f.write_text(txt)
    print(f"   enum: added {line.strip()}")


def patch_valid_list():
    f = ROOT / "scripts/data/convert_lerobot_to_gear.py"
    txt = f.read_text()
    if f'"{EMB}"' in txt:
        print(f"   converter: '{EMB}' already in VALID_EMBODIMENT_TAGS")
        return
    m = re.search(r"VALID_EMBODIMENT_TAGS\s*=\s*\[", txt)
    if not m:
        print("   !! VALID_EMBODIMENT_TAGS not found; add manually:", EMB)
        return
    txt = txt[: m.end()] + f'\n    "{EMB}",' + txt[m.end():]
    f.write_text(txt)
    print(f"   converter: added '{EMB}' to VALID_EMBODIMENT_TAGS")


def write_dataset_yaml():
    f = ROOT / f"groot/vla/configs/data/dreamzero/{EMB}_relative.yaml"
    rel = "\n".join(f"  - {k}" for k in REL_KEYS)
    content = f"""# @package _global_
defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_

max_state_dim: 64
use_global_metadata: false
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
{rel}
max_chunk_size: 4
dataset_shard_sampling_rate: 0.1
mixture_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec
single_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID

{EMB}_data_root: ???

train_dataset:
  _target_: ${{mixture_dataset_cls}}
  _convert_: object
  mixture_spec:
    - dataset_path:
        {EMB}:
          - ${{{EMB}_data_root}}
      dataset_weight: 1.0
      distribute_weights: true
  dataset_class: ${{single_dataset_cls}}
  all_modality_configs: ${{modality_configs}}
  all_transforms: ${{transforms}}
  metadata_versions: ${{metadata_versions}}
  fps: ${{fps}}
  dataset_kwargs:
    video_backend: decord
    use_global_metadata: ${{use_global_metadata}}
    max_chunk_size: ${{max_chunk_size}}
    relative_action: ${{relative_action}}
    relative_action_keys: ${{relative_action_keys}}
    relative_action_per_horizon: ${{relative_action_per_horizon}}
  mixture_kwargs:
    training: true
    balance_dataset_weights: false
    seed: 42
    shard_sampling_rate: ${{dataset_shard_sampling_rate}}
"""
    f.write_text(content)
    print(f"   wrote {f.relative_to(ROOT)}")


def emit_base_snippet():
    video_keys = "\n".join(f"      - {k}" for k in VIDEO_KEYS)
    state_keys = "\n".join(f"      - {k}" for k in STATE_MKEYS)
    action_keys = "\n".join(f"      - {k}" for k in ACTION_MKEYS)
    state_norm = "\n".join(f"        {k}: q99" for k in STATE_MKEYS)
    action_norm = "\n".join(f"        {k}: q99" for k in ACTION_MKEYS)
    deltas25 = ", ".join(str(i) for i in range(25))
    deltas24 = ", ".join(str(i) for i in range(24))

    snippet = f"""
# ====================== PASTE INTO base_48_wan_fine_aug_relative.yaml ======================
modality_config_{EMB}:
  video:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [{deltas25}]
    eval_delta_indices: [0]
    modality_keys:
{video_keys}
  state:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
{state_keys}
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [{deltas24}]
    modality_keys:
{action_keys}
  language:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - annotation.task

transform_{EMB}:
  _target_: groot.vla.data.transform.ComposedModalityTransform
  transforms:
    - <<: *totensor_cfg
      apply_to: ${{modality_config_{EMB}.video.modality_keys}}
    - <<: *crop_cfg
      apply_to: ${{modality_config_{EMB}.video.modality_keys}}
    - <<: *resize_cfg
      apply_to: ${{modality_config_{EMB}.video.modality_keys}}
    - <<: *color_jitter_cfg
      apply_to: ${{modality_config_{EMB}.video.modality_keys}}
    - <<: *to_numpy_cfg
      apply_to: ${{modality_config_{EMB}.video.modality_keys}}
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${{modality_config_{EMB}.state.modality_keys}}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${{modality_config_{EMB}.state.modality_keys}}
      normalization_modes:
{state_norm}
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${{modality_config_{EMB}.action.modality_keys}}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${{modality_config_{EMB}.action.modality_keys}}
      normalization_modes:
{action_norm}
    - _target_: groot.vla.data.transform.ConcatTransform
      video_concat_order: ${{modality_config_{EMB}.video.modality_keys}}
      state_concat_order: ${{modality_config_{EMB}.state.modality_keys}}
      action_concat_order: ${{modality_config_{EMB}.action.modality_keys}}
    - ${{model_specific_transform}}

# Then EXTEND the existing maps at the bottom of the file (add these lines,
# do NOT create duplicate top-level keys):
#   modality_configs:
#     {EMB}: ${{modality_config_{EMB}}}
#   transforms:
#     {EMB}: ${{transform_{EMB}}}
#   metadata_versions:
#     {EMB}: '0221'
#   fps:
#     {EMB}: {FPS}
# ==========================================================================================
"""
    out = ROOT / f"scripts/_base_snippet_{EMB}.yaml"
    out.write_text(snippet)
    print(snippet)
    print(f">> Snippet also saved to {out}")


def main():
    if not (ROOT / "groot").exists():
        sys.exit(f"DREAMZERO_ROOT={ROOT} has no groot/ — clone the repo first (00_setup.sh).")
    print(f">> Registering embodiment '{EMB}' ({NUM_VIEWS} cameras)")
    patch_enum()
    patch_valid_list()
    write_dataset_yaml()
    emit_base_snippet()
    print("\n>> ACTION REQUIRED: paste the block above into")
    print("   groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml")
    print("   then run scripts/04_train_lora_wan22.sh")


if __name__ == "__main__":
    main()
