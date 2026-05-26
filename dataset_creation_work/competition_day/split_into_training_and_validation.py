#!/usr/bin/env python3
"""
Split the 'train' folder into 'training' and 'validation' folders,
each with 'query' and 'gallery' subfolders.

Layout produced:
    training/
        query/    # 225 images (1 per training actor)
        gallery/  # 5175 images (19 per training actor + 900 sfhq)
    validation/
        query/    # 25 images (1 per validation actor)
        gallery/  # 575 images (19 per validation actor + 100 sfhq)

The original 'train' folder is removed only after all counts verify.
"""

import random
import shutil
from pathlib import Path

random.seed(56)  # reproducible split; remove this line for a fresh random split each run

BASE = Path(".")
TRAIN = BASE / "train"
TRAINING = BASE / "training"
VALIDATION = BASE / "validation"

N_ACTORS_TRAIN, N_ACTORS_VAL = 225, 25
N_SFHQ_TRAIN, N_SFHQ_VAL = 900, 100
IMAGES_PER_ACTOR = 20


def main():
    if not TRAIN.is_dir():
        raise FileNotFoundError(f"'{TRAIN.resolve()}' does not exist")
    if TRAINING.exists() or VALIDATION.exists():
        raise FileExistsError(
            "'training' or 'validation' already exists - remove them first to avoid mixing data."
        )

    subdirs = [d for d in TRAIN.iterdir() if d.is_dir()]
    actor_dirs = sorted(d for d in subdirs if not d.name.startswith("sfhq"))
    sfhq_dirs = sorted(d for d in subdirs if d.name.startswith("sfhq"))

    if len(actor_dirs) != 250:
        raise RuntimeError(f"Expected 250 actor folders, found {len(actor_dirs)}")
    if len(sfhq_dirs) != 1000:
        raise RuntimeError(f"Expected 1000 sfhq folders, found {len(sfhq_dirs)}")

    random.shuffle(actor_dirs)
    random.shuffle(sfhq_dirs)

    train_actors, val_actors = actor_dirs[:N_ACTORS_TRAIN], actor_dirs[N_ACTORS_TRAIN:]
    train_sfhq, val_sfhq = sfhq_dirs[:N_SFHQ_TRAIN], sfhq_dirs[N_SFHQ_TRAIN:]

    for parent in (TRAINING, VALIDATION):
        (parent / "query").mkdir(parents=True)
        (parent / "gallery").mkdir(parents=True)

    # Actor images: 1 random image into query, the other 19 into gallery
    for split_dir, actors in ((TRAINING, train_actors), (VALIDATION, val_actors)):
        for actor_dir in actors:
            images = [p for p in actor_dir.iterdir()
                      if p.is_file() and p.suffix.lower() == ".jpg"]
            if len(images) != IMAGES_PER_ACTOR:
                raise RuntimeError(
                    f"'{actor_dir.name}' has {len(images)} JPGs, expected {IMAGES_PER_ACTOR}"
                )
            random.shuffle(images)
            shutil.copy2(images[0], split_dir / "query" / images[0].name)
            for img in images[1:]:
                shutil.copy2(img, split_dir / "gallery" / img.name)

    # SFHQ images: all go into the gallery of their split
    for split_dir, sfhq_folders in ((TRAINING, train_sfhq), (VALIDATION, val_sfhq)):
        for sfhq_dir in sfhq_folders:
            for img in sfhq_dir.iterdir():
                if img.is_file() and img.suffix.lower() == ".jpg":
                    shutil.copy2(img, split_dir / "gallery" / img.name)

    # Verify before deleting the original
    expected = {
        TRAINING / "query":   N_ACTORS_TRAIN,
        TRAINING / "gallery": N_ACTORS_TRAIN * (IMAGES_PER_ACTOR - 1) + N_SFHQ_TRAIN,
        VALIDATION / "query":   N_ACTORS_VAL,
        VALIDATION / "gallery": N_ACTORS_VAL * (IMAGES_PER_ACTOR - 1) + N_SFHQ_VAL,
    }
    print("Final counts:")
    for path, want in expected.items():
        got = sum(1 for _ in path.iterdir())
        status = "OK" if got == want else "MISMATCH"
        print(f"  {path}: {got} (expected {want}) [{status}]")
        if got != want:
            raise RuntimeError(f"Count mismatch in {path} - leaving 'train' in place.")

    print("Removing original 'train' folder...")
    shutil.rmtree(TRAIN)
    print("Done.")


if __name__ == "__main__":
    main()