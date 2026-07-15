#!/usr/bin/env python3
"""Batched SAM 3 text-prompt inference for all images in a directory."""

import re
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.visualization_utils import plot_results


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = Path("/mnt/nas/chenchao/code/sam3/test_data")
OUTPUT_DIR = Path("/mnt/nas/chenchao/code/sam3/outputs/batch")
BPE_PATH = Path(
    "/mnt/nas/chenchao/code/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
)
CHECKPOINT_PATH = Path("/mnt/data/wulan/chenchao/ckpts/sam3.1/sam3.pt")
TEXT_PROMPTS = ["barcode"]
# TEXT_PROMPTS = ["plastic bag", "brown box", "anything with white label"]
BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_name(text: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("_")
    return name or "empty_prompt"


def create_datapoint(image: Image.Image, prompts: list[str], counter: int):
    width, height = image.size
    datapoint = Datapoint(
        find_queries=[],
        images=[SAMImage(data=image, objects=[], size=[height, width])],
    )
    query_ids = []
    for prompt in prompts:
        query_id = counter
        counter += 1
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    # PostProcessImage expects spatial sizes in [height, width].
                    original_size=[height, width],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        query_ids.append(query_id)
    return datapoint, query_ids, counter


def batched(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    if not TEXT_PROMPTS:
        raise ValueError("TEXT_PROMPTS must contain at least one prompt")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        device=str(DEVICE),
        checkpoint_path=str(CHECKPOINT_PATH),
    )
    model.eval()

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=CONFIDENCE_THRESHOLD,
        to_cpu=False,
    )

    query_counter = 1
    saved = 0
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    progress = tqdm(
        total=len(image_paths) * len(TEXT_PROMPTS),
        desc="SAM3 inference",
        unit="result",
    )
    for batch_index, batch_paths in enumerate(
        batched(image_paths, BATCH_SIZE), start=1
    ):
        datapoints = []
        query_records = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            datapoint, query_ids, query_counter = create_datapoint(
                image, TEXT_PROMPTS, query_counter
            )
            datapoints.append(transform(datapoint))
            query_records.extend(
                (query_id, image_path, prompt)
                for query_id, prompt in zip(query_ids, TEXT_PROMPTS)
            )

        batch = collate(datapoints, dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, DEVICE, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            processed_results = postprocessor.process_results(
                output, batch.find_metadatas
            )

        for query_id, image_path, prompt in query_records:
            output_path = (
                OUTPUT_DIR / safe_name(prompt) / f"{image_path.stem}.png"
            )
            with Image.open(image_path) as image:
                plot_results(
                    image.convert("RGB"),
                    processed_results[query_id],
                    output_path=output_path,
                )
            saved += 1
            progress.update(1)

        del batch, output, processed_results
        print(
            f"Finished batch {batch_index}/{total_batches}: "
            f"{len(batch_paths)} image(s)"
        )

    progress.close()
    print(
        f"Saved {saved} visualization(s) for {len(image_paths)} image(s) "
        f"and {len(TEXT_PROMPTS)} prompt(s) to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
