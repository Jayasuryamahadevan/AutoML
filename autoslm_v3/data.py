"""Dataset hygiene primitives designed to prevent train/test leakage."""
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"\s+", " ", value).strip()


def content_hash(example: Dict[str, object]) -> str:
    canonical = normalize_text(str(example.get("instruction", ""))) + "\n" + normalize_text(str(example.get("response", "")))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class DatasetSplit:
    train: List[Dict[str, object]]
    validation: List[Dict[str, object]]
    test: List[Dict[str, object]]
    duplicates_removed: int
    fingerprint: str

    def write_jsonl(self, directory: str) -> None:
        from pathlib import Path
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", self.train), ("validation", self.validation), ("test", self.test)):
            path = root / (name + ".jsonl")
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_dataset(
    examples: Sequence[Dict[str, object]],
    test_fraction: float = 0.1,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """Deduplicate, then split deterministically; group_id stays in one split."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test + validation fractions must be less than 1")

    unique: List[Dict[str, object]] = []
    seen = set()
    for raw in examples:
        row = dict(raw)
        if not str(row.get("instruction", "")).strip() or not str(row.get("response", "")).strip():
            continue
        digest = content_hash(row)
        if digest in seen:
            continue
        seen.add(digest)
        row["_autoslm_content_hash"] = digest
        unique.append(row)

    if len(unique) < 3:
        raise ValueError("at least three unique, non-empty examples are required")

    groups = {}
    for index, row in enumerate(unique):
        group = str(row.get("group_id") or row["_autoslm_content_hash"] or index)
        groups.setdefault(group, []).append(row)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    train, validation, test = [], [], []
    total = len(unique)
    test_target = max(1, int(total * test_fraction))
    validation_target = max(1, int(total * validation_fraction))
    for key in keys:
        group_rows = groups[key]
        if len(test) < test_target:
            test.extend(group_rows)
        elif len(validation) < validation_target:
            validation.extend(group_rows)
        else:
            train.extend(group_rows)

    if not train or not validation or not test:
        raise ValueError("group-aware split produced an empty partition; add examples or reduce group granularity")

    partition_hashes = [
        {str(row["_autoslm_content_hash"]) for row in part}
        for part in (train, validation, test)
    ]
    if (partition_hashes[0] & partition_hashes[1]
            or partition_hashes[0] & partition_hashes[2]
            or partition_hashes[1] & partition_hashes[2]):
        raise RuntimeError("internal error: content leakage across partitions")

    fingerprint = hashlib.sha256("".join(sorted(seen)).encode("ascii")).hexdigest()
    return DatasetSplit(train, validation, test, len(examples) - len(unique), fingerprint)
