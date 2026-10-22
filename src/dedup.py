"""Find duplicate images across and within the train/test splits.

PlantDoc is internet-scraped, so the same photo can appear in several
classes or splits. Exact copies are found by MD5; near-copies (re-encoded,
resized) by a 64-bit difference hash with a small Hamming distance.

    python -m src.dedup            # report only
    python -m src.dedup --move     # move test-side duplicates to data/test_removed/ and
                                   # label-conflicting train duplicates to data/train_removed/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

NEAR_DUP_DISTANCE = 4
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=float)
    return (a[:, 1:] > a[:, :-1]).flatten()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def find_cross_duplicates(train: list[Path], test: list[Path], max_dist: int = NEAR_DUP_DISTANCE):
    """For each test image, its closest training image if within max_dist."""
    train_md5 = {md5(p): p for p in train}
    train_h = np.array([dhash(p) for p in train])
    dups = []
    for p in test:
        exact = train_md5.get(md5(p))
        dist = (train_h != dhash(p)).sum(1)
        j = int(dist.argmin())
        if exact is not None or dist[j] <= max_dist:
            match = exact or train[j]
            dups.append({"test": str(p), "train": str(match),
                         "distance": 0 if exact else int(dist[j]), "exact": exact is not None,
                         "same_class": match.parent.name == p.parent.name})
    return dups


def train_groups(train: list[Path], max_dist: int = NEAR_DUP_DISTANCE) -> np.ndarray:
    """Group ids so near-duplicates inside the training set share a group (for grouped CV)."""
    h = np.array([dhash(p) for p in train])
    parent = list(range(len(train)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(train)):
        close = np.nonzero((h[i + 1:] != h[i]).sum(1) <= max_dist)[0] + i + 1
        for j in close:
            parent[find(int(j))] = find(i)
    roots = [find(i) for i in range(len(train))]
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def label_conflicts(train: list[Path], groups: np.ndarray) -> list[list[Path]]:
    """Duplicate groups whose copies carry different class labels (dataset label noise)."""
    out = []
    for g in np.unique(groups):
        members = [p for p, gi in zip(train, groups) if gi == g]
        if len({p.parent.name for p in members}) > 1:
            out.append(members)
    return out


def _move(src: Path, root: Path):
    dst = root / src.parent.name / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--move", action="store_true", help="move test-side duplicates out of data/test")
    args = ap.parse_args()

    data = Path(args.data)
    train, test = list_images(data / "train"), list_images(data / "test")
    dups = find_cross_duplicates(train, test)
    groups = train_groups(train)
    n_groups_dup = int((np.bincount(groups) > 1).sum())
    conflicts = label_conflicts(train, groups)

    print(f"train={len(train)} test={len(test)}")
    print(f"test images duplicated in train: {len(dups)} "
          f"({sum(d['exact'] for d in dups)} exact, {sum(not d['same_class'] for d in dups)} with a different label)")
    print(f"near-duplicate groups inside train: {n_groups_dup} "
          f"({len(conflicts)} with conflicting labels, {sum(map(len, conflicts))} images)")

    report = {"max_distance": NEAR_DUP_DISTANCE, "n_train": len(train), "n_test": len(test),
              "cross_split_duplicates": dups, "train_duplicate_groups": n_groups_dup,
              "train_label_conflicts": [[str(p) for p in g] for g in conflicts]}
    Path("experiments").mkdir(exist_ok=True)
    # --move records what was removed; a plain run is a post-cleanup check.
    out = "experiments/duplicates.json" if args.move else "experiments/duplicates_check.json"
    Path(out).write_text(json.dumps(report, indent=2))

    if args.move:
        for d in dups:
            _move(Path(d["test"]), data / "test_removed")
        for p in (p for g in conflicts for p in g):
            _move(p, data / "train_removed")
        print(f"moved {len(dups)} test images to {data / 'test_removed'} and "
              f"{sum(map(len, conflicts))} train images to {data / 'train_removed'}")


if __name__ == "__main__":
    main()
