"""
Dataset Reorganizer (no Spark required)
========================================
Replaces the PySpark step in download_RDD_split.py with plain Python.
Reads from dataset_extracted/ and writes the same structure to dataset_final/.

Usage:
    python reorganize_dataset.py
"""

import os
import shutil
import hashlib
import random

random.seed(42)

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR  = os.path.join(SCRIPT_DIR, "dataset_extracted")
FINAL_SPLIT_DIR= os.path.join(SCRIPT_DIR, "dataset_final")


def process_and_split_dataset(source_dir, output_dir):
    # ── Collect all files ──────────────────────────────────────────────────────
    print("[*] Scanning extracted files...")
    all_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if not f.startswith('.'):
                all_files.append(os.path.join(root, f))
    print(f"[->] Found {len(all_files):,} files.")

    # ── Build celebrity identity → split map ──────────────────────────────────
    print("[*] Building identity → split map...")
    celebrity_ids = set()
    for path in all_files:
        fname = os.path.basename(path)
        if '_' in fname and not fname.startswith(('c', 's')):
            parts = fname.split('_')
            if parts[0].isdigit():
                celebrity_ids.add(parts[0])

    celebrity_ids = list(celebrity_ids)
    random.shuffle(celebrity_ids)
    n_total = len(celebrity_ids)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)

    identity_split_map = {}
    for i, cid in enumerate(celebrity_ids):
        if i < n_train:
            identity_split_map[cid] = 'train'
        elif i < n_train + n_val:
            identity_split_map[cid] = 'val'
        else:
            identity_split_map[cid] = 'test'

    print(f"[->] {n_total} identities → "
          f"train={n_train}, val={n_val}, test={n_total-n_train-n_val}")

    # ── Copy files into final structure ───────────────────────────────────────
    print("[*] Sorting files into dataset_final/ ...")
    skipped = 0
    for i, path in enumerate(all_files):
        if i % 5000 == 0:
            print(f"    {i:,}/{len(all_files):,}...")

        fname = os.path.basename(path)

        if fname.startswith('c'):
            hash_score = int(hashlib.md5(fname.encode()).hexdigest(), 16) % 100
            split      = 'train' if hash_score < 80 else ('val' if hash_score < 90 else 'test')
            dest_dir   = os.path.join(output_dir, split, 'cartoons')

        elif fname.startswith('s'):
            hash_score = int(hashlib.md5(fname.encode()).hexdigest(), 16) % 100
            split      = 'train' if hash_score < 80 else ('val' if hash_score < 90 else 'test')
            dest_dir   = os.path.join(output_dir, split, 'synthetic')

        elif '_' in fname:
            identity_id = fname.split('_')[0]
            if identity_id.isdigit():
                split    = identity_split_map.get(identity_id, 'test')
                dest_dir = os.path.join(output_dir, split, 'celebrities', identity_id)
            else:
                skipped += 1
                continue
        else:
            skipped += 1
            continue

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dest_dir, fname))

    print(f"[+] Done! Skipped {skipped} unrecognised files.")
    print(f"[+] Output → {output_dir}/")


if __name__ == "__main__":
    if not os.path.exists(EXTRACTED_DIR):
        print(f"[!] dataset_extracted/ not found at: {EXTRACTED_DIR}")
        print("    Run download_RDD_split.py first (it will fail on Spark but the")
        print("    extraction step succeeds). Then run this script.")
    else:
        process_and_split_dataset(EXTRACTED_DIR, FINAL_SPLIT_DIR)
        print("\n[*] Structure created:")
        for split in ('train', 'val', 'test'):
            split_dir = os.path.join(FINAL_SPLIT_DIR, split)
            if os.path.exists(split_dir):
                subdirs = os.listdir(split_dir)
                counts = {}
                for sd in subdirs:
                    sd_path = os.path.join(split_dir, sd)
                    if os.path.isdir(sd_path):
                        # Count recursively for celebrities (has sub-dirs per identity)
                        count = sum(len(fs) for _, _, fs in os.walk(sd_path))
                        counts[sd] = count
                print(f"  {split}/: {counts}")
