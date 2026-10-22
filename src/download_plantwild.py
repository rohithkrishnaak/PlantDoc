"""Download Tomato and Potato classes from PlantWild and PlantWild_v2 datasets.

Target source: https://tqwei05.github.io/PlantWild/
Hosted on Hugging Face: https://huggingface.co/datasets/uqtwei2/PlantWild

Downloads only Tomato and Potato classes using optimized HTTP range requests
to download only the contiguous byte blocks belonging to the target classes,
saving gigabytes of unnecessary downloads.
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path


class HttpRangeReader(io.RawIOBase):
    """Seekable stream that reads ranges from a remote HTTP file on demand."""
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            self._size = int(resp.headers['Content-Length'])
        self._pos = 0

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self._pos
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET: self._pos = offset
        elif whence == io.SEEK_CUR: self._pos += offset
        elif whence == io.SEEK_END: self._pos = self._size + offset
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size == -1: size = self._size - self._pos
        if self._pos >= self._size or size <= 0: return b''
        end = min(self._pos + size - 1, self._size - 1)
        req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={self._pos}-{end}'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self._pos += len(data)
        return data


def fetch_byte_range(url: str, start: int, end: int, max_retries: int = 5) -> bytes:
    """Download a byte range [start, end] with retries and progress logging."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = bytearray()
                expected = end - start + 1
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    data.extend(chunk)
                if len(data) == expected:
                    return bytes(data)
                print(f"  [warn] Incomplete read ({len(data)}/{expected}), retrying...")
        except Exception as e:
            print(f"  [warn] Range fetch failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch range {start}-{end} after {max_retries} attempts.")


def extract_group(url: str, group: list[zipfile.ZipInfo], out_base: Path, prefix_strip: str = ""):
    """Download and extract a cluster of zip entries."""
    start = group[0].header_offset
    end = max(x.header_offset + 30 + 1024 + x.compress_size for x in group)
    size_mb = (end - start + 1) / (1024 * 1024)
    print(f"  Downloading group ({len(group)} files, {size_mb:.2f} MB)...", flush=True)
    t0 = time.time()
    raw_bytes = fetch_byte_range(url, start, end)
    dur = time.time() - t0
    speed = size_mb / dur if dur > 0 else 0
    print(f"  Fetched {size_mb:.2f} MB in {dur:.1f}s ({speed:.2f} MB/s). Decompressing files...", flush=True)

    extracted_count = 0
    for info in group:
        offset_in_buf = info.header_offset - start
        if offset_in_buf < 0 or offset_in_buf + 30 > len(raw_bytes):
            print(f"    [error] Header offset out of range for {info.filename}")
            continue

        # Parse local file header
        magic, ver, flag, method, time_val, date_val, crc, csize, usize, nlen, elen = struct.unpack(
            '<IHHHHHIIIHH', raw_bytes[offset_in_buf : offset_in_buf + 30]
        )
        if magic != 0x04034B50:
            print(f"    [error] Invalid magic 0x{magic:x} for {info.filename}")
            continue

        data_start = offset_in_buf + 30 + nlen + elen
        data_end = data_start + csize
        cdata = raw_bytes[data_start:data_end]

        try:
            if method == 8:  # Deflate
                decompressed = zlib.decompress(cdata, -15)
            elif method == 0:  # Stored
                decompressed = cdata
            else:
                print(f"    [error] Unsupported method {method} for {info.filename}")
                continue

            # Verify CRC
            actual_crc = zlib.crc32(decompressed)
            if actual_crc != info.CRC:
                print(f"    [error] CRC mismatch for {info.filename}: expected {info.CRC}, got {actual_crc}")
                continue

            # Relative path within target dir
            rel_name = info.filename
            if prefix_strip and rel_name.startswith(prefix_strip):
                rel_name = rel_name[len(prefix_strip):]
            rel_name = rel_name.lstrip('/\\')

            target_file = out_base / rel_name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_bytes(decompressed)
            extracted_count += 1
        except Exception as e:
            print(f"    [error] Decompressing {info.filename}: {e}")

    print(f"  Successfully extracted {extracted_count}/{len(group)} files.", flush=True)


def download_dataset(url: str, out_dir: Path, dataset_name: str, prefix_strip: str = "", gap_threshold: int = 10 * 1024 * 1024):
    print(f"\n=======================================================")
    print(f"Starting {dataset_name} download to {out_dir}")
    print(f"=======================================================", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading central directory from remote archive...", flush=True)
    reader = HttpRangeReader(url)
    zf = zipfile.ZipFile(reader)

    # Filter target files
    target_entries = []
    meta_entries = []
    for info in zf.infolist():
        fname_lower = info.filename.lower()
        if info.is_dir():
            continue
        if 'tomato' in fname_lower or 'potato' in fname_lower:
            target_entries.append(info)
        elif not fname_lower.endswith('.jpg') and not fname_lower.endswith('.png') and not fname_lower.endswith('/'):
            meta_entries.append(info)

    target_entries.sort(key=lambda x: x.header_offset)
    print(f"Found {len(target_entries)} Tomato/Potato images.")
    if meta_entries:
        print(f"Found {len(meta_entries)} metadata files: {[m.filename for m in meta_entries]}")

    # Extract metadata files first
    for m in meta_entries:
        try:
            rel_name = m.filename
            if prefix_strip and rel_name.startswith(prefix_strip):
                rel_name = rel_name[len(prefix_strip):]
            rel_name = rel_name.lstrip('/\\')
            target_file = out_dir / rel_name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            content = zf.read(m)
            target_file.write_bytes(content)
            print(f"  Saved metadata: {rel_name} ({len(content)} bytes)")
        except Exception as e:
            print(f"  [warn] Could not extract metadata {m.filename}: {e}")

    # Group into contiguous chunks
    groups = []
    curr = [target_entries[0]]
    for item in target_entries[1:]:
        prev_end = curr[-1].header_offset + 30 + 1024 + curr[-1].compress_size
        if item.header_offset - prev_end < gap_threshold:
            curr.append(item)
        else:
            groups.append(curr)
            curr = [item]
    groups.append(curr)

    total_mb = sum((max(x.header_offset + 30 + 1024 + x.compress_size for x in g) - g[0].header_offset) / (1024*1024) for g in groups)
    print(f"Partitioned {len(target_entries)} files into {len(groups)} range requests (total {total_mb:.2f} MB).", flush=True)

    for i, g in enumerate(groups, 1):
        print(f"\nProcessing group {i}/{len(groups)}:")
        extract_group(url, g, out_dir, prefix_strip=prefix_strip)

    print(f"\nFinished extracting {dataset_name} to {out_dir}!")


def summarize_folder(folder: Path):
    print(f"\nSummary of contents in {folder}:")
    classes = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            cls_name = p.parent.name
            classes[cls_name] = classes.get(cls_name, 0) + 1
    total_imgs = sum(classes.values())
    for cls, cnt in sorted(classes.items()):
        print(f"  - {cls}: {cnt} images")
    print(f"Total: {len(classes)} classes, {total_imgs} images.")


def main():
    base_data = Path("data")

    # 1. PlantWild (v1)
    # The original PlantWild archive has prefix 'plantwild/'
    v1_url = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"
    v1_dir = base_data / "plantwild"
    download_dataset(v1_url, v1_dir, "PlantWild (v1)", prefix_strip="plantwild")
    summarize_folder(v1_dir)

    # 2. PlantWild_v2
    # The v2 archive has prefix 'plantwild_v2/'
    v2_url = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild_v2.zip"
    v2_dir = base_data / "plantwild_v2"
    download_dataset(v2_url, v2_dir, "PlantWild_v2", prefix_strip="plantwild_v2")
    summarize_folder(v2_dir)


if __name__ == "__main__":
    main()
