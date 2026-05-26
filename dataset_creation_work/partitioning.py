import os
import re
import random
import shutil
from collections import defaultdict

random.seed(555)  # reproducibility

# --- Paths ---
BASE_DIR = "."
OUT_DIR  = "retrieval_strategists_full_dataset_in_partitions_v[version_number_to_change_manually]"

CELEB_DIR  = "CelebA_Subset"
TOON_DIR   = "Cartoon_Faces_Subset"
SYNTH_DIR  = "Synthetic_Faces_Subset"

# --- Build directory tree ---
for split in ["training", "validation", "test"]:
    for role in ["query", "gallery"]:
        os.makedirs(os.path.join(OUT_DIR, split, role), exist_ok=True)

# ---------------------------------------------------------------
# 1.  Group celebrity images by identity
# ---------------------------------------------------------------
celeb_groups = defaultdict(list)  # identity_str -> [full_path, ...]

pattern = re.compile(r'^(\d+)_\d+\.jpg$')

for fname in os.listdir(CELEB_DIR):
    m = pattern.match(fname)
    if not m:
        print(f"  WARNING: unexpected filename skipped: {fname}")
        continue
    identity = m.group(1)
    celeb_groups[identity].append(os.path.join(CELEB_DIR, fname))

# Sanity check: every identity must have exactly 4 images
for ident, paths in celeb_groups.items():
    if len(paths) != 4:
        raise ValueError(f"Identity {ident} has {len(paths)} images (expected 4).")

all_identities = list(celeb_groups.keys())
count = 5000 # IMPORTANT: for v1 this was 1000. Everything else was the same.
if len(all_identities) != count:
    raise ValueError(f"Expected {count} identities, found {len(all_identities)}.")

# ---------------------------------------------------------------
# 2.  Shuffle and split identities  80 / 10 / 10
# ---------------------------------------------------------------
random.shuffle(all_identities)
train_ids = all_identities[:int(count*0.8)]
val_ids   = all_identities[int(count*0.8):int(count*0.9)]
test_ids  = all_identities[int(count*0.9):]

# ---------------------------------------------------------------
# 3.  Shuffle and split non-celebrity images  80 / 10 / 10
# ---------------------------------------------------------------
toon_files  = [os.path.join(TOON_DIR,  f) for f in os.listdir(TOON_DIR)  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
synth_files = [os.path.join(SYNTH_DIR, f) for f in os.listdir(SYNTH_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(synth_files) != count: raise ValueError(f"Expected {count} synthetics, found {len(synth_files)}.")
if len(toon_files)  != count: raise ValueError(f"Expected {count} cartoons,   found {len(toon_files)}.")

random.shuffle(toon_files)
random.shuffle(synth_files)

toon_splits  = {"training": toon_files[:int(count*0.8)],  "validation": toon_files[int(count*0.8):int(count*0.9)],  "test": toon_files[int(count*0.9):]}
synth_splits = {"training": synth_files[:int(count*0.8)], "validation": synth_files[int(count*0.8):int(count*0.9)], "test": synth_files[int(count*0.9):]}

id_splits = {"training": train_ids, "validation": val_ids, "test": test_ids}

# ---------------------------------------------------------------
# 4.  Copy files into the right partition / role directories
# ---------------------------------------------------------------
def copy_to(src_path, split, role):
    dst = os.path.join(OUT_DIR, split, role, os.path.basename(src_path))
    shutil.copy2(src_path, dst)

for split, identities in id_splits.items():
    print(f"\nProcessing {split}...")

    # -- Celebrity images --
    for ident in identities:
        paths = celeb_groups[ident][:]      # list of 4 paths
        query_path = random.choice(paths)   # pick 1 at random as the query
        gallery_paths = [p for p in paths if p != query_path]

        copy_to(query_path, split, "query")
        for p in gallery_paths:
            copy_to(p, split, "gallery")

    # -- Cartoons and synthetics go entirely to gallery --
    for p in toon_splits[split]:
        copy_to(p, split, "gallery")
    for p in synth_splits[split]:
        copy_to(p, split, "gallery")

    # -- Summary --
    n_query   = len(os.listdir(os.path.join(OUT_DIR, split, "query")))
    n_gallery = len(os.listdir(os.path.join(OUT_DIR, split, "gallery")))
    print(f"  query:   {n_query}")
    print(f"  gallery: {n_gallery}")
    print(f"  total:   {n_query + n_gallery}")

print("\nAll done.")