from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    source_dataset_hub: str = "ljt019/rust-review-merged"
    backup_dataset_hub: str = "ljt019/rust-review-hq-merged-backup"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = BackupConfig()

    logger.info("Downloading dataset '%s'", config.source_dataset_hub)
    try:
        dataset = load_dataset(config.source_dataset_hub, split="train", streaming=False)
    except Exception as exc:
        logger.error("Failed to download dataset '%s': %s", config.source_dataset_hub, exc)
        return

    assert isinstance(dataset, Dataset)

    logger.info(
        "Pushing backup with %s examples to '%s'",
        len(dataset),
        config.backup_dataset_hub,
    )
    dataset.push_to_hub(config.backup_dataset_hub)
    logger.info("Backup complete")


if __name__ == "__main__":
    main()
