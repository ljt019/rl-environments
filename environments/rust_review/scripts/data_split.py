from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    source_dataset_hub: str = "ljt019/rust-review-merged-final-w-ds"
    split_dataset_hub: str = "ljt019/rust-review-final-on-god"
    test_size: int = 250
    shuffle_seed: int = 42


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = SplitConfig()

    logger.info("Downloading dataset '%s'", config.source_dataset_hub)
    try:
        dataset = load_dataset(config.source_dataset_hub, split="train", streaming=False)
    except Exception as exc:
        logger.error("Failed to download dataset '%s': %s", config.source_dataset_hub, exc)
        return

    assert isinstance(dataset, Dataset)

    logger.info("Original dataset size: %s examples", len(dataset))

    # Shuffle thoroughly
    logger.info("Shuffling dataset with seed %s", config.shuffle_seed)
    shuffled_dataset = dataset.shuffle(seed=config.shuffle_seed)

    # Split into test and train
    if len(shuffled_dataset) < config.test_size:
        logger.error("Dataset too small (%s) to split %s examples for test", len(shuffled_dataset), config.test_size)
        return

    test_dataset = shuffled_dataset.select(range(config.test_size))
    train_dataset = shuffled_dataset.select(range(config.test_size, len(shuffled_dataset)))

    logger.info(
        "Split dataset: %s test examples, %s train examples",
        len(test_dataset),
        len(train_dataset),
    )

    # Create DatasetDict with train/test splits
    split_datasets = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    logger.info(
        "Pushing split dataset to '%s'",
        config.split_dataset_hub,
    )
    split_datasets.push_to_hub(config.split_dataset_hub)
    logger.info("Split upload complete")


if __name__ == "__main__":
    main()
