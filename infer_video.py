"""
Minimal SAM3 video demo.

Edit the constants below, then run:
    python demo.py

The script reads an MP4, runs text-prompt video segmentation on the first
INFERENCE_SECONDS seconds, and writes an overlay visualization MP4.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
import torch


# ============================= User settings =============================

# Input MP4.
INPUT_MP4 = "/mnt/nas/chenchao/code/Depth-Anything-3/cc_demo/val_dexhand_16eps_10hz/videos/observation.images.camera_head_color/chunk-000/file-000.mp4"

# Segment objects matching these text prompts. Each prompt is run in its own
# session and the results are merged for visualization.
TEXT_PROMPTS = ["plastic bag", "brown box"]

# Only run inference on the first N seconds of INPUT_MP4.
INFERENCE_SECONDS = 60.0

# Visualization output path. Parent directories are created automatically.
VIS_OUTPUT_PATH = "outputs/demo_vis.mp4"

# SAM3 version. Use "sam3.1" for the newer object-multiplex predictor.
SAM3_VERSION = "sam3.1"

# Local checkpoint file or directory. If this is a directory, the script will
# look for the default checkpoint filename for SAM3_VERSION inside it.
CHECKPOINT_PATH = "/mnt/data/wulan/chenchao/ckpts/sam3.1"

# Prompt frame. Usually 0 is fine for a short demo.
PROMPT_FRAME_INDEX = 0

# Output mask probability threshold. This is the closest exposed setting to a
# "tolerance" knob in the predictor API: lower keeps more mask pixels/objects,
# higher is stricter.
OUTPUT_PROB_THRESH = 0.5

# Propagation controls: "both", "forward", or "backward".
PROPAGATION_DIRECTION = "both"

# Limit tracking frames after prompting. None means use all extracted frames.
MAX_FRAME_NUM_TO_TRACK = None

# Set True if you want model compilation/warmup. False starts faster.
COMPILE_MODEL = False

# FlashAttention 3 is optional. Keep this False unless your environment has
# flash_attn_interface installed.
USE_FA3 = False

# Needed by SAM3.1 multiplex mode; should be >= expected objects in the video.
MAX_NUM_OBJECTS = 64

# Overlay style.
MASK_ALPHA = 0.45
DRAW_OBJECT_IDS = True

# VSCode's video preview is happier with H.264/yuv420p than OpenCV's mp4v.
TRANSCODE_VIS_TO_H264 = True

# Temporary extracted frame directory. None uses a system temp directory.
FRAME_DIR = None

# ========================================================================


MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 64, 128),
    (128, 255, 0),
    (64, 128, 255),
]

DEFAULT_CHECKPOINT_FILENAMES = {
    "sam3.1": ("sam3.1_multiplex.pt", "sam3.1.pt"),
    "sam3": ("sam3.pt",),
}

MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024


def extract_mp4_frames(video_path: Path, frame_dir: Path, seconds: float) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    max_frames = max(1, int(round(seconds * fps)))
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while frame_count < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_path = frame_dir / f"{frame_count:05d}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    return frame_count, fps


def masks_from_response(response: dict) -> dict[int, np.ndarray]:
    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks")
    if binary_masks is None:
        return {}

    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.detach().cpu().numpy()
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.detach().cpu().numpy()

    masks = {}
    for index, obj_id in enumerate(obj_ids):
        mask = binary_masks[index]
        if mask.ndim == 3:
            mask = mask[0]
        masks[int(obj_id)] = mask.astype(bool)
    return masks


def collect_video_masks(model, session_id: str) -> dict[int, dict[int, np.ndarray]]:
    masks_by_frame = {}
    for response in model.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": PROPAGATION_DIRECTION,
            "max_frame_num_to_track": MAX_FRAME_NUM_TO_TRACK,
            "output_prob_thresh": OUTPUT_PROB_THRESH,
        }
    ):
        frame_index = response.get("frame_index")
        if frame_index is None:
            continue
        masks_by_frame[int(frame_index)] = masks_from_response(response)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return masks_by_frame


def offset_mask_ids(
    masks_by_frame: dict[int, dict[int, np.ndarray]], offset: int
) -> dict[int, dict[int, np.ndarray]]:
    return {
        frame_idx: {obj_id + offset: mask for obj_id, mask in masks.items()}
        for frame_idx, masks in masks_by_frame.items()
    }


def merge_video_masks(
    base: dict[int, dict[int, np.ndarray]],
    extra: dict[int, dict[int, np.ndarray]],
) -> dict[int, dict[int, np.ndarray]]:
    for frame_idx, masks in extra.items():
        base.setdefault(frame_idx, {}).update(masks)
    return base


def overlay_masks(frame_bgr: np.ndarray, masks_by_obj: dict[int, np.ndarray]) -> np.ndarray:
    overlay = frame_bgr.astype(np.float32).copy()

    for obj_id, mask in sorted(masks_by_obj.items()):
        color_rgb = MASK_COLORS[obj_id % len(MASK_COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        for channel, value in enumerate(color_bgr):
            overlay[:, :, channel] = np.where(
                mask,
                overlay[:, :, channel] * (1.0 - MASK_ALPHA) + value * MASK_ALPHA,
                overlay[:, :, channel],
            )

        if DRAW_OBJECT_IDS and mask.any():
            ys, xs = np.where(mask)
            center = (int(xs.mean()), int(ys.mean()))
            cv2.putText(
                overlay,
                str(obj_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def write_visualization(
    frame_dir: Path,
    output_path: Path,
    fps: float,
    frame_count: int,
    masks_by_frame: dict[int, dict[int, np.ndarray]],
) -> None:
    first_frame = cv2.imread(str(frame_dir / "00000.jpg"))
    if first_frame is None:
        raise RuntimeError(f"Could not read first extracted frame in: {frame_dir}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path
    if TRANSCODE_VIS_TO_H264 and output_path.suffix.lower() == ".mp4":
        raw_output_path = output_path.with_name(f"{output_path.stem}_opencv_tmp.mp4")

    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {raw_output_path}")

    try:
        for frame_index in range(frame_count):
            frame_path = frame_dir / f"{frame_index:05d}.jpg"
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue
            vis = overlay_masks(frame_bgr, masks_by_frame.get(frame_index, {}))
            writer.write(vis)
    finally:
        writer.release()

    if raw_output_path != output_path:
        transcode_to_h264(raw_output_path, output_path)


def transcode_to_h264(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(str(input_path), str(output_path))
        print("ffmpeg not found; kept OpenCV mp4v output, which VSCode may not preview.")
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        shutil.move(str(input_path), str(output_path))
        stderr = exc.stderr.decode("utf-8", errors="replace")
        print("ffmpeg H.264 transcode failed; kept OpenCV mp4v output.")
        print(stderr[-1000:])
    else:
        input_path.unlink(missing_ok=True)


def resolve_checkpoint_path(checkpoint_path: str, version: str) -> str:
    path = Path(checkpoint_path)
    candidate_names = DEFAULT_CHECKPOINT_FILENAMES.get(version)
    if candidate_names is None:
        raise ValueError(f"Unsupported SAM3_VERSION: {version!r}")

    if path.is_file():
        return str(path)

    if path.is_dir():
        for name in candidate_names:
            candidate = path / name
            if candidate.is_file():
                return str(candidate)
        expected = ", ".join(str(path / name) for name in candidate_names)
        raise FileNotFoundError(f"No checkpoint file found. Expected one of: {expected}")

    # Convenience for directory-like paths that may exist only in the runtime
    # environment where this demo is launched.
    if path.suffix == "":
        return str(path / candidate_names[0])

    raise FileNotFoundError(f"CHECKPOINT_PATH does not exist: {path}")


def validate_checkpoint_path(checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is not a file: {path}")

    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(512)

    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "Checkpoint looks like a Git LFS pointer, not real weights. "
            f"Run `git lfs pull` or redownload the checkpoint: {path}"
        )

    stripped = prefix.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise RuntimeError(
            "Checkpoint looks like an HTML/XML page, not real weights. "
            "This usually means the download failed or Hugging Face auth/access is missing. "
            f"Redownload after `hf auth login`: {path}"
        )

    if size < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"Checkpoint is unexpectedly small ({size} bytes): {path}. "
            "It is probably incomplete or not the real .pt file. Redownload it."
        )


def build_predictor():
    from sam3 import build_sam3_predictor

    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH, SAM3_VERSION)
    validate_checkpoint_path(checkpoint_path)

    build_kwargs = {
        "version": SAM3_VERSION,
        "compile": COMPILE_MODEL,
        "async_loading_frames": False,
        "checkpoint_path": checkpoint_path,
        "use_fa3": USE_FA3,
    }
    if SAM3_VERSION == "sam3.1":
        build_kwargs["warm_up"] = COMPILE_MODEL
        build_kwargs["max_num_objects"] = MAX_NUM_OBJECTS
        build_kwargs["default_output_prob_thresh"] = OUTPUT_PROB_THRESH

    return build_sam3_predictor(**build_kwargs)


def patch_start_session_for_current_backend(predictor) -> None:
    """Keep the README start_session flow, but filter init_state kwargs locally."""

    def start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type

        sig = inspect.signature(self.model.init_state)
        valid_params = set(sig.parameters.keys())
        init_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_params}

        inference_state = self.model.init_state(**init_kwargs)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = MethodType(start_session, predictor)


def run_text_prompt(model, frame_dir: Path, text_prompt: str) -> dict[int, dict[int, np.ndarray]]:
    response = model.start_session(resource_path=str(frame_dir))
    session_id = response["session_id"]
    model.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": PROMPT_FRAME_INDEX,
            "text": text_prompt,
            "output_prob_thresh": OUTPUT_PROB_THRESH,
        }
    )
    return collect_video_masks(model, session_id)


def main() -> None:
    input_mp4 = Path(INPUT_MP4)
    output_path = Path(VIS_OUTPUT_PATH)
    if not input_mp4.exists():
        raise FileNotFoundError(f"INPUT_MP4 does not exist: {input_mp4}")

    temp_root = None
    if FRAME_DIR is None:
        temp_root = Path(tempfile.mkdtemp(prefix="sam3_demo_"))
        frame_dir = temp_root / "frames"
    else:
        frame_dir = Path(FRAME_DIR)
        if frame_dir.exists():
            shutil.rmtree(frame_dir)

    try:
        frame_count, fps = extract_mp4_frames(input_mp4, frame_dir, INFERENCE_SECONDS)

        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        model = build_predictor()
        patch_start_session_for_current_backend(model)

        t0 = time.perf_counter()
        masks_by_frame = {}
        prompt_counts = {}
        for prompt_index, text_prompt in enumerate(TEXT_PROMPTS):
            prompt_masks = run_text_prompt(model, frame_dir, text_prompt)
            prompt_counts[text_prompt] = len(prompt_masks.get(PROMPT_FRAME_INDEX, {}))
            prompt_masks = offset_mask_ids(prompt_masks, offset=prompt_index * 1000)
            merge_video_masks(masks_by_frame, prompt_masks)
        elapsed = time.perf_counter() - t0

        write_visualization(frame_dir, output_path, fps, frame_count, masks_by_frame)

        frame0_objects = len(masks_by_frame.get(PROMPT_FRAME_INDEX, {}))
        print(f"Input: {input_mp4}")
        print(f"Checkpoint: {resolve_checkpoint_path(CHECKPOINT_PATH, SAM3_VERSION)}")
        print(f"Prompts: {TEXT_PROMPTS!r}")
        print(f"Objects on prompt frame by prompt: {prompt_counts}")
        print(f"Output probability threshold: {OUTPUT_PROB_THRESH}")
        print(f"Frames: {frame_count} ({frame_count / fps:.2f}s at {fps:.2f} FPS)")
        print(f"Detected objects on prompt frame: {frame0_objects}")
        print(f"Inference time: {elapsed:.2f}s")
        print(f"Visualization: {output_path}")
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
