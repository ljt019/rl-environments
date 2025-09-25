from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    source_dataset_1: str = "ljt019/rust-review-merged-final-w-codst"
    source_dataset_2: str = "ljt019/rust-review-ds"
    merged_dataset_hub: str = "ljt019/rust-review-merged-final-w-ds"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = MergeConfig()

    logger.info("Downloading first dataset '%s'", config.source_dataset_1)
    try:
        dataset1 = load_dataset(config.source_dataset_1, split="train", streaming=False)
    except Exception as exc:
        logger.error("Failed to download dataset '%s': %s", config.source_dataset_1, exc)
        return

    assert isinstance(dataset1, Dataset)

    logger.info("Downloading second dataset '%s'", config.source_dataset_2)
    try:
        dataset2 = load_dataset(config.source_dataset_2, split="train", streaming=False)
    except Exception as exc:
        logger.error("Failed to download dataset '%s': %s", config.source_dataset_2, exc)
        return

    assert isinstance(dataset2, Dataset)

    logger.info(
        "Merging datasets: %s examples + %s examples = %s total",
        len(dataset1),
        len(dataset2),
        len(dataset1) + len(dataset2),
    )

    merged_dataset = concatenate_datasets([dataset1, dataset2])

    logger.info(
        "Pushing merged dataset with %s examples to '%s'",
        len(merged_dataset),
        config.merged_dataset_hub,
    )
    merged_dataset.push_to_hub(config.merged_dataset_hub)
    logger.info("Merge complete")


if __name__ == "__main__":
    main()
