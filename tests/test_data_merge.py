"""Tests for the PlantWild merge (leakage checks), the GPU logistic regression and evaluation outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate import TEST_DIR, output_paths
from src.gpu_logreg import TorchLogisticRegression
from src.merge_plantwild import CLASS_MAP, DHASH_DIST, MANIFEST, components, same_photo_edges
from src.model import CLASSES

ROOT = Path(__file__).resolve().parent.parent


def test_gpu_logreg_matches_sklearn():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 20)) * 2
    y = rng.integers(0, 4, 400)
    y[:40] = 0  # imbalance, so class_weight="balanced" matters
    X = StandardScaler().fit_transform(centers[y] + rng.normal(size=(400, 20)))
    for C in (0.01, 1.0):
        sk = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, y)
        tl = TorchLogisticRegression(C=C, device="cpu").fit(X, y)
        assert np.abs(sk.predict_proba(X) - tl.predict_proba(X)).max() < 0.02
        assert (sk.predict(X) == tl.predict(X)).mean() > 0.99


def test_same_photo_edges_and_components():
    md5s = ["a", "a", "b", "c", "d"]
    hashes = np.zeros((5, 64), bool)  # 0, 1, 2 share a dHash
    hashes[3, :32] = True  # 3 and 4: far from everyone in dHash
    hashes[4, 32:] = True
    emb = np.eye(5)
    emb[3] = emb[4] = 0
    emb[3, 3] = emb[4, 3] = 1.0  # 3 and 4: identical embeddings (a crop)
    edges = {(a, b) for a, b, _ in same_photo_edges(md5s, hashes, emb)}
    assert (0, 1) in edges          # same MD5
    assert (0, 2) in edges          # identical dHash (all zeros), distance 0 <= DHASH_DIST
    assert (3, 4) in edges          # cosine 1.0
    assert (2, 3) not in edges and (2, 4) not in edges
    comp = components(5, same_photo_edges(md5s, hashes, emb), keep=set(range(5)))
    assert comp[0] == comp[1] == comp[2] and comp[3] == comp[4] and comp[0] != comp[3]


def test_class_map_targets_are_model_classes():
    assert set(CLASS_MAP.values()) == set(CLASSES)


def test_eval_output_paths_suffixed_for_other_test_sets():
    assert output_paths(TEST_DIR)[0].name == "evaluation_report.json"
    report, plot, cm, csv = output_paths(Path("data/test_plantwild"))
    assert report.name == "evaluation_report_plantwild.json"
    assert plot.name == "coverage_curve_plantwild.png" and cm.name == "confusion_matrix_plantwild.png"
    assert csv.name == "test_predictions_plantwild.csv"


def test_merge_manifest_has_no_train_test_overlap():
    if not (ROOT / MANIFEST).exists():
        return  # run python -m src.merge_plantwild first
    man = json.loads((ROOT / MANIFEST).read_text())
    train = set(man["train_groups"])
    test = set(man["pw_test"])
    assert not train & test
    leak = man["summary"]["leak_check"]
    assert leak["md5_overlap"] == 0
    assert leak["min_train_test_dhash"] > DHASH_DIST
    assert leak["max_train_test_cos"] < man["params"]["cos_sim"]
    # no source photo is placed in both splits (PlantWild reuses numeric names across classes)
    rel = lambda p: (Path(p).parent.name, Path(p).name)
    assert not {rel(p) for p in test} & {rel(p) for p in train}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
