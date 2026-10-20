"""Data retrieval, parsing, duplicate-aware splits, and character transforms."""

from __future__ import annotations

import random
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def download_dataset(archive_path: Path, url: str) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "sms-spam-course-project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as out:
            out.write(response.read())
    return archive_path


def load_messages(archive_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
        lines = archive.read(member).decode("utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        if "\t" not in line:
            continue
        label, text = line.split("\t", maxsplit=1)
        if label in {"ham", "spam"}:
            rows.append((label, text))
    frame = pd.DataFrame(rows, columns=["label", "text"])
    if len(frame) != 5574:
        raise ValueError(f"Expected 5,574 SMS records, found {len(frame)}")
    frame["group"] = frame["text"].str.casefold().str.strip()
    return frame


def make_grouped_split(frame: pd.DataFrame, seed: int, train_fraction: float,
                       validation_fraction: float, test_fraction: float):
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1")
    # Stratification is applied to one representative row per normalized text group.
    group_frame = frame.drop_duplicates("group").reset_index(drop=True)
    train_groups, remaining = train_test_split(
        group_frame, train_size=train_fraction, random_state=seed,
        stratify=group_frame["label"],
    )
    val_share = validation_fraction / (validation_fraction + test_fraction)
    val_groups, test_groups = train_test_split(
        remaining, train_size=val_share, random_state=seed + 1,
        stratify=remaining["label"],
    )
    grouped = [set(part["group"]) for part in (train_groups, val_groups, test_groups)]
    partitions = [frame[frame["group"].isin(group_set)].drop(columns="group").reset_index(drop=True)
                  for group_set in grouped]
    return tuple(partitions)


def augment_message(text: str, rng: random.Random, probability: float) -> str:
    chars = list(text)
    if not chars:
        return text
    if rng.random() < probability:
        positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if positions:
            i = rng.choice(positions)
            chars[i] = str(rng.randrange(10))
    if rng.random() < probability:
        positions = [i for i, ch in enumerate(chars) if ch.isalpha() and i > 0]
        if positions:
            i = rng.choice(positions)
            chars.insert(i, " ")
    return "".join(chars)


def obfuscate_message(text: str, seed: int) -> str:
    return augment_message(text, random.Random(seed), 0.8)
