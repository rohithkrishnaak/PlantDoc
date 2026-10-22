"""Merge PlantWild tomato/potato photos into the PlantDoc training set without leakage.

    python -m src.merge_plantwild            # writes data/train_merged/ and data/test_plantwild/

PlantWild (https://tqwei05.github.io/PlantWild/) is also web-scraped, so it
shares photos with PlantDoc, including PlantDoc *test* photos. A difference
hash only catches re-encodes, so "same photo" here means any of:
  - identical MD5
  - dHash Hamming distance <= DHASH_DIST
  - DINOv2-S cosine similarity >= COS_SIM (catches crops / re-edits; unrelated
    photos of the same disease stay below ~0.85)
Rules use direct same-photo pairs (PlantDoc train photos are resolved first):
  1. same photo as a PlantDoc test photo -> excluded from training
  2. same photo carries a different label anywhere (other dataset, other class,
     PlantWild_v2 relabel; non-target classes such as septoria count) -> dropped
  3. PlantWild photo that is a near-identical copy (dHash or cos >= 0.95) of a
     photo already kept -> dropped as redundant
PlantWild_v2 is a relabelled subset of v1 (all of its photos are copies of v1
photos), so it contributes labels for check 2 but no new photos.

Kept photos are then chained into components; 15% of the PlantWild-only components (stratified, seeded) become a
second held-out test set, data/test_plantwild. The near-duplicate group of every
training photo is written to experiments/merge_manifest.json so grouped CV can
keep copies in one fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src import mlflow_utils as mlfu
from src.dedup import dhash, list_images, md5

DHASH_DIST = 6
COS_SIM = 0.90
STRONG_COS_SIM = 0.95  # near-identical: a redundant copy, not just a re-crop
PW_TEST_FRAC = 0.15
SEED = 42
EMBED_MODEL = "vit_small_patch14_dinov2.lvd142m"
MANIFEST = Path("experiments/merge_manifest.json")

# PlantWild class folder -> PlantDoc class. "potato leaf" (healthy potato) has no
# PlantDoc class, so it is used only for duplicate / label-conflict checks.
CLASS_MAP = {
    "potato early blight": "Potato Early Blight",
    "potato late blight": "Potato Late Blight",
    "tomato bacterial leaf spot": "Tomato Bacterial Spot",
    "tomato early blight": "Tomato Early Blight",
    "tomato late blight": "Tomato Late Blight",
    "tomato leaf": "Tomato Healthy",
}


def embed(paths: list[Path], batch: int = 64) -> np.ndarray:
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(EMBED_MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(dev)
            out.append(torch.nn.functional.normalize(m(x), dim=1).float().cpu().numpy())
    return np.concatenate(out)


def same_photo_edges(md5s: list[str], hashes: np.ndarray, emb: np.ndarray):
    """Pairs (i, j, strong) judged to be the same photo; strong = near-identical copy."""
    by_md5 = defaultdict(list)
    for i, h in enumerate(md5s):
        by_md5[h].append(i)
    edges = {(min(g[0], j), max(g[0], j)): True for g in by_md5.values() for j in g[1:]}
    for i in range(len(md5s)):
        dh = (hashes[i + 1:] != hashes[i]).sum(1)
        cos = emb[i + 1:] @ emb[i]
        for j in np.nonzero((dh <= DHASH_DIST) | (cos >= COS_SIM))[0]:
            key = (i, int(j) + i + 1)
            edges[key] = edges.get(key, False) or bool(dh[j] <= DHASH_DIST or cos[j] >= STRONG_COS_SIM)
    return [(a, b, s) for (a, b), s in edges.items()]


def components(n: int, edges, keep: set[int]) -> np.ndarray:
    """Connected components over `keep` (transitive), so chained copies share a group."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, _ in edges:
        if a in keep and b in keep:
            parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n)])


def place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # same volume: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)

    # (path, source, label); label is the PlantDoc class, or "pw:<folder>" for non-target classes
    items = [(p, "pd_train", p.parent.name) for p in list_images(data / "train")]
    items += [(p, "pd_test", p.parent.name) for p in list_images(data / "test")]
    items += [(p, "pw1", CLASS_MAP.get(p.parent.name, f"pw:{p.parent.name}"))
              for p in list_images(data / "plantwild" / "images")]
    items += [(p, "pw2", CLASS_MAP.get(p.parent.name, f"pw:{p.parent.name}"))
              for p in list_images(data / "plantwild_v2")]
    print(f"hashing + embedding {len(items)} images ...", flush=True)
    paths = [p for p, _, _ in items]
    md5s = [md5(p) for p in paths]
    hashes = np.array([dhash(p) for p in paths])
    emb = embed(paths)
    edges = same_photo_edges(md5s, hashes, emb)
    nbrs = defaultdict(list)
    for a, b, s in edges:
        nbrs[a].append((b, s))
        nbrs[b].append((a, s))

    src = [s for _, s, _ in items]
    lab = [l for _, _, l in items]
    reasons, dropped = Counter(), []

    def drop(i, why):
        reasons[f"{why} ({src[i]})"] += 1
        dropped.append((i, why))

    # Rules apply to direct same-photo pairs (no chaining), PlantDoc train first so
    # a PlantWild copy of a PlantDoc photo is the one dropped as redundant.
    order = sorted((i for i in range(len(items)) if src[i] in ("pd_train", "pw1")),
                   key=lambda i: (src[i] != "pd_train", str(paths[i])))
    kept: set[int] = set()
    for i in order:
        if any(src[j] == "pd_test" for j, _ in nbrs[i]):
            drop(i, "same photo as a PlantDoc test photo")
        elif any(lab[j] != lab[i] for j, _ in nbrs[i]):
            drop(i, "label conflict")  # includes v1 photos relabelled in v2
        elif lab[i].startswith("pw:"):
            continue  # non-target PlantWild class
        elif src[i] == "pw1" and any(s and j in kept for j, s in nbrs[i]):
            drop(i, "near-identical copy of a kept photo")
        else:
            kept.add(i)

    comp = components(len(items), edges, kept)
    members = defaultdict(list)
    for i in kept:
        members[comp[i]].append(i)

    # Second held-out test set: whole PlantWild-only components, stratified by class,
    # so no test photo has a same-photo partner in training even through a chain.
    rng = np.random.default_rng(SEED)
    by_cls = defaultdict(list)
    for c, idx in members.items():
        if all(src[i] == "pw1" for i in idx):
            by_cls[lab[idx[0]]].append(c)
    n_pw = Counter(lab[i] for i in kept if src[i] == "pw1")
    test_comps = set()
    for cls in sorted(by_cls):
        comps = sorted(by_cls[cls])
        rng.shuffle(comps)
        target, got = PW_TEST_FRAC * n_pw[cls], 0
        for c in comps:
            if got >= target:
                break
            test_comps.add(c)
            got += len(members[c])
    keep_train = sorted((i, int(comp[i])) for i in kept if comp[i] not in test_comps)
    pw_test = sorted((i, int(comp[i])) for i in kept if comp[i] in test_comps)

    out_train, out_test = data / "train_merged", data / "test_plantwild"
    for d in (out_train, out_test):
        if d.exists():
            shutil.rmtree(d)
    manifest = {"params": {"dhash_dist": DHASH_DIST, "cos_sim": COS_SIM, "embed_model": EMBED_MODEL,
                           "pw_test_frac": PW_TEST_FRAC, "seed": SEED},
                "train_groups": {}, "pw_test": [], "dropped": []}
    for i, c in keep_train:
        p, src, label = items[i]
        prefix = "pd" if src == "pd_train" else "pw"
        dst = out_train / label / f"{prefix}__{p.name}"
        place(p, dst)
        manifest["train_groups"][dst.as_posix()] = int(c)
    for i, c in pw_test:
        p, _, label = items[i]
        dst = out_test / label / f"pw__{p.name}"
        place(p, dst)
        manifest["pw_test"].append(dst.as_posix())
    manifest["dropped"] = [{"path": str(paths[i]), "source": items[i][1], "label": items[i][2], "reason": r}
                           for i, r in dropped]

    # Verify: no training photo is the same photo as any photo in either test set.
    tr_idx = [i for i, _ in keep_train]
    te_idx = [i for i, s in enumerate(items) if s[1] == "pd_test"] + [i for i, _ in pw_test]
    max_cos = float((emb[te_idx] @ emb[tr_idx].T).max())
    min_dh = int(min((hashes[tr_idx] != hashes[j]).sum(1).min() for j in te_idx))
    md5_overlap = len({md5s[i] for i in tr_idx} & {md5s[j] for j in te_idx})
    assert max_cos < COS_SIM and min_dh > DHASH_DIST and md5_overlap == 0, (max_cos, min_dh, md5_overlap)

    counts_train = Counter(items[i][2] for i, _ in keep_train)
    counts_src = Counter(items[i][1] for i, _ in keep_train)
    counts_test = Counter(items[i][2] for i, _ in pw_test)
    summary = {"n_input": len(items), "n_train": len(keep_train), "train_by_source": dict(counts_src),
               "train_by_class": dict(sorted(counts_train.items())), "n_pw_test": len(pw_test),
               "pw_test_by_class": dict(sorted(counts_test.items())), "removal_reasons": dict(reasons),
               "leak_check": {"max_train_test_cos": max_cos, "min_train_test_dhash": min_dh,
                              "md5_overlap": md5_overlap}}
    manifest["summary"] = summary
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))

    with mlfu.mlflow_run(run_name="data-merge-plantwild", enabled=not args.no_mlflow,
                         tags={"stage": "data", "dataset": "PlantDoc+PlantWild"}):
        mlfu.log_params(manifest["params"])
        mlfu.log_metrics({"n_train": len(keep_train), "n_pw_test": len(pw_test),
                          "n_train_plantdoc": counts_src.get("pd_train", 0),
                          "n_train_plantwild": counts_src.get("pw1", 0),
                          "leak_max_train_test_cos": max_cos, "leak_min_train_test_dhash": min_dh})
        mlfu.log_metrics({f"n_train_{k.lower().replace(' ', '_')}": v for k, v in counts_train.items()})
        mlfu.log_dict(summary, "data/merge_summary.json")
        mlfu.log_artifact(MANIFEST, artifact_path="data")
        try:
            import mlflow
            # Dataset lineage: content hash of the file list
            digest = hashlib.md5("|".join(sorted(manifest["train_groups"])).encode()).hexdigest()[:12]
            mlflow.set_tag("train_merged_digest", digest)
        except Exception:
            pass


if __name__ == "__main__":
    main()
