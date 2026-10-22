"""Stress-test the PlantWild dataset before training on it.

PlantWild (Wei et al., ACM MM 2024; CC BY-NC-ND 4.0) is 18,542 web-scraped
field photos in 89 class folders. Download and unzip it so that the layout is
<root>/images/<class name>/*.jpg (the layout the authors' MVPDR loader expects).

    python -m experiments.plantwild_stress_test --root plantwild
    python -m experiments.plantwild_stress_test --root plantwild --plantdoc data --baseline

Reports, and writes to experiments/plantwild_report.json:
  * images per class, imbalance ratio, potato/tomato classes and counts
  * unreadable, tiny and non-RGB files
  * exact (MD5) and near (dHash) duplicates, inside a class and across classes
  * test images that also sit in train, for the split being used
  * PlantWild images that are copies of our PlantDoc train/test photos (--plantdoc)
  * --baseline: linear probe on frozen ImageNet features for the potato and
    tomato classes, scored on the PlantWild test split and on the PlantDoc test set
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.dedup import NEAR_DUP_DISTANCE, IMAGE_EXTS

MIN_SIDE = 64

# PlantWild folder name (lower-case) -> our class name. Folder names are matched
# loosely, so check the "mapped" section of the report.
OUR_CLASSES = {
    "Potato Early Blight": ("potato early blight", "potato leaf early blight"),
    "Potato Late Blight": ("potato late blight", "potato leaf late blight"),
    "Potato Healthy": ("potato leaf", "potato healthy"),
    "Tomato Bacterial Spot": ("tomato bacterial spot", "tomato leaf bacterial spot"),
    "Tomato Early Blight": ("tomato early blight", "tomato early blight leaf"),
    "Tomato Late Blight": ("tomato late blight", "tomato leaf late blight"),
    "Tomato Healthy": ("tomato leaf", "tomato healthy"),
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def inspect(path: Path) -> tuple[np.ndarray | None, str | None, tuple[int, int] | None, str | None]:
    """dHash, MD5, size and mode, or (None, md5, None, error) for an unreadable file."""
    digest = md5(path)
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            mode, size = im.mode, im.size
            a = np.asarray(im.convert("L").resize((9, 8)), dtype=float)
        return (a[:, 1:] > a[:, :-1]).flatten(), digest, size, mode
    except Exception as e:  # noqa: BLE001 - any decode failure counts as corrupt
        return None, digest, None, f"{type(e).__name__}: {e}"


def list_dataset(root: Path) -> dict[str, list[Path]]:
    image_dir = root / "images" if (root / "images").is_dir() else root
    return {c.name: sorted(p for p in c.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            for c in sorted(image_dir.iterdir()) if c.is_dir()}


def load_split(root: Path, items: list[tuple[Path, str]], seed: int) -> tuple[list[str], str]:
    """'train'/'val'/'test' per item.

    Uses the authors' trainval.txt (lines "path=label=domain", domain 1=train,
    2=val, 0=test) when it ships with the download. Otherwise makes the same
    per-class 70/10/20 random split the authors' loader makes; theirs depends on
    os.listdir order, so it is not reproducible across machines.
    """
    txt = root / "trainval.txt"
    if txt.exists():
        by_rel = {}
        for line in txt.read_text().splitlines():
            name, _, domain = line.rsplit("=", 2)
            parts = Path(name).parts
            by_rel["/".join(parts[-2:])] = {"0": "test", "1": "train", "2": "val"}[domain]
        split = [by_rel.get(f"{p.parent.name}/{p.name}", "missing") for p, _ in items]
        return split, f"authors' {txt.name}"

    rng = random.Random(seed)
    split = ["test"] * len(items)
    by_class = defaultdict(list)
    for i, (_, c) in enumerate(items):
        by_class[c].append(i)
    for idxs in by_class.values():
        trainval = rng.sample(idxs, int(np.ceil(len(idxs) * 0.8)))
        cut = int(np.ceil(len(trainval) * 0.875))
        for i in trainval[:cut]:
            split[i] = "train"
        for i in trainval[cut:]:
            split[i] = "val"
    return split, f"per-class 70/10/20 random split, seed {seed}"


def union_find_groups(hashes: np.ndarray, max_dist: int) -> np.ndarray:
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    packed = np.packbits(hashes, axis=1)
    for i in range(len(hashes)):
        x = np.unpackbits(packed[i + 1:] ^ packed[i], axis=1).sum(1)
        for j in np.nonzero(x <= max_dist)[0] + i + 1:
            parent[find(int(j))] = find(i)
    return np.array([find(i) for i in range(len(hashes))])


def nearest(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and Hamming distance to the closest ref hash, per query hash."""
    q, r = np.packbits(query, axis=1), np.packbits(ref, axis=1)
    idx, dist = np.zeros(len(q), int), np.zeros(len(q), int)
    for i in range(len(q)):
        d = np.unpackbits(r ^ q[i], axis=1).sum(1)
        idx[i], dist[i] = int(d.argmin()), int(d.min())
    return idx, dist


def map_classes(classes: list[str]) -> dict[str, str]:
    lower = {c.lower().replace("_", " ").strip(): c for c in classes}
    out = {}
    for ours, names in OUR_CLASSES.items():
        hit = next((lower[n] for n in names if n in lower), None)
        if hit:
            out[ours] = hit
    return out


def plantdoc_overlap(pw_paths, pw_hashes, pw_md5, plantdoc: Path, max_dist: int) -> dict:
    out = {}
    for split in ("train", "test"):
        files = sorted(p for p in (plantdoc / split).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            continue
        info = [inspect(p) for p in files]
        ok = [i for i, x in enumerate(info) if x[0] is not None]
        hashes = np.array([info[i][0] for i in ok])
        idx, dist = nearest(hashes, pw_hashes)
        md5s = set(pw_md5)
        hits = [{"plantdoc": str(files[k]), "plantwild": str(pw_paths[idx[n]]), "distance": int(dist[n]),
                 "exact": info[k][1] in md5s}
                for n, k in enumerate(ok) if dist[n] <= max_dist or info[k][1] in md5s]
        out[split] = {"n": len(files), "found_in_plantwild": len(hits), "matches": hits}
    return out


def baseline(items, split, classes_map, plantdoc: Path | None, arch: str) -> dict:
    """Frozen ImageNet backbone + logistic regression on the mapped classes."""
    import timm
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    from src.model import PREPROCESS

    model = timm.create_model(arch, pretrained=True, num_classes=0).eval()
    rev = {v: k for k, v in classes_map.items()}

    @torch.no_grad()
    def embed(paths):
        feats = []
        for i in range(0, len(paths), 32):
            batch = torch.stack([PREPROCESS(Image.open(p).convert("RGB")) for p in paths[i:i + 32]])
            feats.append(model(batch).numpy())
        return np.concatenate(feats)

    sel = {s: [(p, rev[c]) for (p, c), sp in zip(items, split) if c in rev and sp == s]
           for s in ("train", "val", "test")}
    names = sorted(classes_map)
    xs = {s: embed([p for p, _ in v]) for s, v in sel.items() if v}
    ys = {s: np.array([names.index(c) for _, c in v]) for s, v in sel.items() if v}
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(xs["train"], ys["train"])

    def score(x, y):
        pred = clf.predict_proba(x)
        top1 = pred.argmax(1)
        top3 = np.argsort(-pred, 1)[:, :3]
        return {"n": int(len(y)), "top1": round(accuracy_score(y, top1), 4),
                "balanced_top1": round(balanced_accuracy_score(y, top1), 4),
                "top3": round(float(np.mean([t in row for t, row in zip(y, top3)])), 4)}

    res = {"arch": arch, "classes": names,
           "train_acc": score(xs["train"], ys["train"]),
           "plantwild_test": score(xs["test"], ys["test"])}
    if plantdoc is not None and (plantdoc / "test").is_dir():
        pd_items = [(p, c.name) for c in sorted((plantdoc / "test").iterdir()) if c.is_dir() and c.name in names
                    for p in sorted(c.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if pd_items:
            x = embed([p for p, _ in pd_items])
            res["plantdoc_test"] = score(x, np.array([names.index(c) for _, c in pd_items]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plantwild", help="folder holding images/<class>/")
    ap.add_argument("--plantdoc", default=None, help="our PlantDoc data/ folder, to check overlap")
    ap.add_argument("--max-dist", type=int, default=NEAR_DUP_DISTANCE)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--out", default="experiments/plantwild_report.json")
    args = ap.parse_args()

    root = Path(args.root)
    data = list_dataset(root)
    items = [(p, c) for c, ps in data.items() for p in ps]
    print(f"{len(items)} images in {len(data)} classes")

    info = [inspect(p) for p, _ in items]
    corrupt = [{"path": str(items[i][0]), "error": x[3]} for i, x in enumerate(info) if x[0] is None]
    ok = [i for i, x in enumerate(info) if x[0] is not None]
    tiny = [str(items[i][0]) for i in ok if min(info[i][2]) < MIN_SIDE]
    modes = Counter(info[i][3] for i in ok)

    counts = {c: len(ps) for c, ps in data.items()}
    nonzero = [n for n in counts.values() if n]
    potato = {c: n for c, n in counts.items() if "potato" in c.lower()}
    tomato = {c: n for c, n in counts.items() if "tomato" in c.lower()}
    classes_map = map_classes(list(data))

    # Exact duplicates by MD5, near duplicates by dHash.
    by_md5 = defaultdict(list)
    for i in ok:
        by_md5[info[i][1]].append(i)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    hashes = np.array([info[i][0] for i in ok])
    roots = union_find_groups(hashes, args.max_dist)
    near = defaultdict(list)
    for k, r in zip(ok, roots):
        near[r].append(k)
    near_groups = [g for g in near.values() if len(g) > 1]

    def describe(groups):
        cross = [g for g in groups if len({items[i][1] for i in g}) > 1]
        return {"groups": len(groups), "images": sum(map(len, groups)),
                "redundant_copies": sum(len(g) - 1 for g in groups),
                "groups_spanning_classes": len(cross),
                "examples_spanning_classes": [[f"{items[i][1]}/{items[i][0].name}" for i in g] for g in cross[:25]]}

    split, split_source = load_split(root, items, args.seed)
    split_counts = Counter(split)
    leaks = []
    for g in near_groups:
        sp = {split[i] for i in g}
        if "test" in sp and sp & {"train", "val"}:
            leaks.append(g)
    test_leaked = sum(1 for g in leaks for i in g if split[i] == "test")

    report = {
        "root": str(root.resolve()), "n_images": len(items), "n_classes": len(data),
        "per_class": counts,
        "imbalance": {"max": max(nonzero), "min": min(nonzero), "ratio": round(max(nonzero) / min(nonzero), 2),
                      "median": float(np.median(nonzero)), "classes_under_50": sorted(c for c, n in counts.items() if n < 50)},
        "potato_classes": potato, "tomato_classes": tomato, "mapped_to_our_classes": classes_map,
        "corrupt": corrupt, "tiny_images": tiny, "image_modes": dict(modes),
        "exact_duplicates": describe(exact_groups),
        "near_duplicates": {"max_hamming": args.max_dist, **describe(near_groups)},
        "split": {"source": split_source, "counts": dict(split_counts),
                  "test_images_with_copy_in_train_or_val": test_leaked,
                  "test_share_leaked": round(test_leaked / max(split_counts["test"], 1), 4),
                  "leaked_in_potato_tomato": sum(1 for g in leaks for i in g
                                                 if split[i] == "test" and items[i][1] in classes_map.values())},
    }
    if args.plantdoc:
        report["plantdoc_overlap"] = plantdoc_overlap(
            [items[i][0] for i in ok], hashes, [info[i][1] for i in ok], Path(args.plantdoc), args.max_dist)
    if args.baseline:
        report["baseline"] = baseline([items[i] for i in ok], [split[i] for i in ok], classes_map,
                                      Path(args.plantdoc) if args.plantdoc else None, args.arch)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    im = report["imbalance"]
    print(f"per class: min {im['min']}, median {im['median']:.0f}, max {im['max']} (ratio {im['ratio']})")
    print(f"potato classes: {potato}")
    print(f"mapped to our classes: {classes_map}")
    print(f"corrupt: {len(corrupt)}, smaller than {MIN_SIDE}px: {len(tiny)}, modes: {dict(modes)}")
    for k in ("exact_duplicates", "near_duplicates"):
        d = report[k]
        print(f"{k}: {d['groups']} groups, {d['redundant_copies']} redundant copies, "
              f"{d['groups_spanning_classes']} groups span >1 class")
    s = report["split"]
    print(f"split ({s['source']}): {s['counts']}; test images with a copy in train/val: "
          f"{s['test_images_with_copy_in_train_or_val']} ({s['test_share_leaked']:.1%}), "
          f"{s['leaked_in_potato_tomato']} in potato/tomato")
    for sp, d in report.get("plantdoc_overlap", {}).items():
        print(f"PlantDoc {sp}: {d['found_in_plantwild']} of {d['n']} images also in PlantWild")
    if "baseline" in report:
        b = report["baseline"]
        print(f"baseline {b['arch']}: train {b['train_acc']}, PlantWild test {b['plantwild_test']}, "
              f"PlantDoc test {b.get('plantdoc_test')}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
