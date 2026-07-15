#!/usr/bin/env python3
"""Extract configurable frames from every episode in one or more datasets."""

import random
from pathlib import Path

import cv2
import pyarrow.parquet as pq


# -----------------------------------------------------------------------------
# Inputs: edit these five constants.
# Every DATA_ROOTS entry must have the same relative file layout.
# -----------------------------------------------------------------------------
DATA_ROOTS = [
    "/mnt/data/wulan/yuyong/data/i7/logistics/val_dexhand_16eps_10hz",
]
OUTPUT_DIR = Path("/mnt/nas/chenchao/code/sam3/test_data")
INCLUDE_FIRST_FRAME = True
INCLUDE_LAST_FRAME = True
NUM_MIDDLE_FRAMES = 3


DATA_PARQUET_RELATIVE = Path("data/chunk-000/file-000.parquet")
EPISODE_META_RELATIVE = Path("meta/episodes/chunk-000/file-000.parquet")
VIDEO_RELATIVE = Path(
    "videos/observation.images.camera_head_color/chunk-000/file-000.mp4"
)
VIDEO_KEY = "observation.images.camera_head_color"
RANDOM_SEED = 42


def load_episode_rows(data_parquet: Path) -> dict[int, list[tuple[int, float]]]:
    data = pq.read_table(
        data_parquet, columns=["episode_index", "frame_index", "timestamp"]
    ).to_pydict()
    rows_by_episode: dict[int, list[tuple[int, float]]] = {}
    for episode, frame_index, timestamp in zip(
        data["episode_index"], data["frame_index"], data["timestamp"]
    ):
        rows_by_episode.setdefault(int(episode), []).append(
            (int(frame_index), float(timestamp))
        )
    return rows_by_episode


def load_episode_starts(episode_meta_path: Path) -> dict[int, tuple[int, float]]:
    start_column = f"videos/{VIDEO_KEY}/from_timestamp"
    episode_meta = pq.read_table(
        episode_meta_path, columns=["episode_index", "length", start_column]
    ).to_pylist()
    return {
        int(row["episode_index"]): (int(row["length"]), float(row[start_column]))
        for row in episode_meta
    }


def select_positions(length: int, rng: random.Random) -> list[int]:
    available_middle = length - 2
    if NUM_MIDDLE_FRAMES > available_middle:
        raise ValueError(
            f"Cannot sample {NUM_MIDDLE_FRAMES} middle frames from an episode "
            f"with {length} total frames"
        )

    positions = []
    if INCLUDE_FIRST_FRAME:
        positions.append(0)
    if NUM_MIDDLE_FRAMES:
        positions.extend(
            sorted(rng.sample(range(1, length - 1), NUM_MIDDLE_FRAMES))
        )
    if INCLUDE_LAST_FRAME:
        positions.append(length - 1)
    return positions


def process_source(
    data_root: Path,
    source_index: int,
    rng: random.Random,
) -> tuple[int, int]:
    data_parquet = data_root / DATA_PARQUET_RELATIVE
    episode_meta_path = data_root / EPISODE_META_RELATIVE
    video_path = data_root / VIDEO_RELATIVE
    for path in (data_parquet, episode_meta_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source file does not exist: {path}")

    rows_by_episode = load_episode_rows(data_parquet)
    starts = load_episode_starts(episode_meta_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0 or video_frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video reports invalid FPS or frame count: {video_path}")

    saved = 0
    source_episode_count = 0
    try:
        for source_episode in sorted(rows_by_episode):
            source_episode_count += 1
            rows = sorted(rows_by_episode[source_episode])
            if source_episode not in starts:
                raise KeyError(
                    f"Missing episode metadata for episode {source_episode} in {data_root}"
                )
            expected_length, video_start_s = starts[source_episode]
            if len(rows) != expected_length:
                raise ValueError(
                    f"Source {data_root}, episode {source_episode}: parquet has "
                    f"{len(rows)} rows, metadata says {expected_length}"
                )

            selected_positions = select_positions(len(rows), rng)
            details = []
            for order, position in enumerate(selected_positions, start=1):
                parquet_frame, local_timestamp_s = rows[position]
                global_timestamp_s = video_start_s + local_timestamp_s
                video_frame = round(global_timestamp_s * video_fps)
                video_frame = min(max(video_frame, 0), video_frame_count - 1)

                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Failed reading source {data_root}, episode "
                        f"{source_episode}, video frame {video_frame}"
                    )

                output_path = (
                    OUTPUT_DIR
                    / f"s{source_index}_eps{source_episode:02d}_{order}.png"
                )
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(f"Failed writing image: {output_path}")
                saved += 1
                details.append(
                    f"{order}:data={parquet_frame},video={video_frame},"
                    f"t={global_timestamp_s:.3f}s"
                )
            print(
                f"source={data_root.name} source_episode={source_episode} "
                f"prefix=s{source_index}: " + "; ".join(details)
            )
    finally:
        cap.release()

    return source_episode_count, saved


def main() -> None:
    if not DATA_ROOTS:
        raise ValueError("DATA_ROOTS must contain at least one dataset root")
    if NUM_MIDDLE_FRAMES < 0:
        raise ValueError("NUM_MIDDLE_FRAMES cannot be negative")
    if not (INCLUDE_FIRST_FRAME or INCLUDE_LAST_FRAME or NUM_MIDDLE_FRAMES > 0):
        raise ValueError("At least one frame must be selected per episode")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    total_episodes = 0
    total_saved = 0
    for source_index, data_root in enumerate(DATA_ROOTS):
        episode_count, saved = process_source(Path(data_root), source_index, rng)
        total_episodes += episode_count
        total_saved += saved

    print(
        f"Saved {total_saved} images from {total_episodes} episodes across "
        f"{len(DATA_ROOTS)} source(s) to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
