"""
Dataset utilities — loading and preparing datasets for training.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_local_dataset(file_path: str) -> list[dict]:
    """
    Load a JSON dataset from disk.

    If the file does not exist, creates a minimal dummy dataset
    so the pipeline doesn't crash during development.
    """
    if not os.path.exists(file_path):
        logger.warning("%s not found — creating dummy dataset", file_path)
        dummy_data = [
            {
                "prompt": "Write a function that adds two numbers.",
                "completion": "def add(a, b):\n    return a + b",
            }
        ]
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=2)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d samples from %s", len(data), file_path)
    return data


def prepare_dataset(
    train_source: str = "data/train.jsonl",
    valid_source: str = "data/valid.jsonl",
) -> tuple[str, str]:
    """
    Load raw datasets, save processed copies, and return their paths.

    Returns:
        (train_file_path, valid_file_path)
    """
    logger.info("Preparing datasets …")
    train_data = load_local_dataset(train_source)
    valid_data = load_local_dataset(valid_source)

    train_file = "processed_train.json"
    valid_file = "processed_valid.json"

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False)

    logger.info("Datasets ready: %s (%d), %s (%d)",
                train_file, len(train_data), valid_file, len(valid_data))
    return train_file, valid_file