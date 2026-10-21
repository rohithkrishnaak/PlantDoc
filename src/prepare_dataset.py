"""Extract and prepare the PlantDoc dataset subset for training and testing.

Extracts 6 target classes from the cloned bare git repository, sanitizes
filenames, validates image integrity, and organizes into standard ImageFolder format:
  data/
    train/
      Potato Early Blight/
      Potato Late Blight/
      Tomato Bacterial Spot/
      Tomato Early Blight/
      Tomato Late Blight/
      Tomato Healthy/
    test/
      ...
"""
from __future__ import annotations

import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from PIL import Image

CLASS_MAPPING = {
    "Potato leaf early blight": "Potato Early Blight",
    "Potato leaf late blight": "Potato Late Blight",
    "Tomato leaf bacterial spot": "Tomato Bacterial Spot",
    "Tomato Early blight leaf": "Tomato Early Blight",
    "Tomato leaf late blight": "Tomato Late Blight",
    "Tomato leaf": "Tomato Healthy",
}

DATA_DIR = Path("data")
BARE_REPO = Path("plantdoc_bare")


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def extract_split(split_name: str):
    print(f"=== Extracting {split_name} split ===")
    for raw_folder, clean_name in CLASS_MAPPING.items():
        tree_path = f"{split_name}/{raw_folder}"
        target_dir = DATA_DIR / split_name / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            output = run_cmd(["git", "-C", str(BARE_REPO), "ls-tree", "-r", "HEAD", tree_path])
        except subprocess.CalledProcessError as exc:
            print(f"Failed to ls-tree for {tree_path}: {exc}")
            continue

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        valid_count = 0
        for idx, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 4:
                continue
            blob_hash = parts[2]
            
            # Fetch blob binary
            fetch_res = subprocess.run(
                ["git", "-C", str(BARE_REPO), "cat-file", "-p", blob_hash],
                capture_output=True
            )
            blob_bytes = fetch_res.stdout
            
            # Validate with PIL
            try:
                img = Image.open(BytesIO(blob_bytes))
                img.verify()
                # Re-open after verify to save
                img = Image.open(BytesIO(blob_bytes)).convert("RGB")
                save_path = target_dir / f"{idx:04d}.jpg"
                img.save(save_path, "JPEG", quality=90)
                valid_count += 1
            except Exception as e:
                # Corrupted or unreadable image
                pass

        print(f"  [{split_name}] {clean_name}: {valid_count} valid images extracted.")


def main():
    if not BARE_REPO.exists():
        print(f"Cloning bare repo from GitHub...")
        subprocess.run([
            "git", "clone", "--bare", "--filter=blob:none", "--depth", "1",
            "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(BARE_REPO)
        ], check=True)

    extract_split("train")
    extract_split("test")
    print("\nDataset extraction complete!")


if __name__ == "__main__":
    main()
