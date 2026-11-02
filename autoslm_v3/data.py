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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(example: Dict[str, object]) -> str:
    canonical = normalize_text(str(example.get("instruction", ""))) + "\n" + normalize_text(str(example.get("response", "")))
    return _digest(canonical)


def prompt_hash(example: Dict[str, object]) -> str:
    """Hash the input independently so repeated prompts cannot cross splits."""
    return _digest(normalize_text(str(example.get("instruction", ""))))


def source_group_id(example: Dict[str, object], fallback: str) -> str:
    """Resolve stable source-document identity from flat or nested provenance."""
    provenance = example.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    value = (
        example.get("group_id")
        or example.get("source_document_id")
        or example.get("document_id")
        or provenance.get("source_document_id")
        or provenance.get("document_id")
    )
    return str(value) if value else fallback


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
    """Deduplicate, then split deterministically; group_id and repeated prompts stay together."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test + validation fractions must be less than 1")

    unique: List[Dict[str, object]] = []
    seen = set()
    duplicates_removed = 0
    for raw in examples:
        row = dict(raw)
        if not str(row.get("instruction", "")).strip() or not str(row.get("response", "")).strip():
            continue
        digest = content_hash(row)
        if digest in seen:
            duplicates_removed += 1
            continue
        seen.add(digest)
        row["_autoslm_content_hash"] = digest
        row["_autoslm_prompt_hash"] = prompt_hash(row)
        unique.append(row)

    if len(unique) < 3:
        raise ValueError("at least three unique, non-empty examples are required")

    # Connected components enforce both document-level grouping and prompt-level
    # grouping. A prompt can connect records from multiple source groups.
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, row in enumerate(unique):
        document_node = "document:" + source_group_id(row, "row:" + str(index))
        prompt_node = "prompt:" + str(row["_autoslm_prompt_hash"])
        union(document_node, prompt_node)

    groups = {}
    for index, row in enumerate(unique):
        document_node = "document:" + source_group_id(row, "row:" + str(index))
        root = find(document_node)
        groups.setdefault(root, []).append(row)

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
    prompt_hashes = [
        {str(row["_autoslm_prompt_hash"]) for row in part}
        for part in (train, validation, test)
    ]
    if any(
        partition_hashes[i] & partition_hashes[j]
        or prompt_hashes[i] & prompt_hashes[j]
        for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("internal error: content or prompt leakage across partitions")

    fingerprint = _digest("".join(sorted(seen)))
    return DatasetSplit(train, validation, test, duplicates_removed, fingerprint)
