import os
import random
import shutil
from collections import defaultdict

# Adjust img_dir if ls showed a different nesting
img_dir = os.path.expanduser("celeba_raw/img_align_celeba/img_align_celeba")
identity_file = os.path.expanduser("identity_CelebA.txt")
output_imgs = os.path.expanduser("CelebA_Subset")

os.makedirs(output_imgs, exist_ok=True)

# 1. Read identities and group images by identity
print("Parsing identity file...")
identity_to_images = defaultdict(list)

with open(identity_file, 'r') as f_in:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) == 2:
            img_name, identity = parts
            identity_to_images[identity].append(img_name)

# 2. Filter for identities that have at least 4 images
valid_identities = [uid for uid, imgs in identity_to_images.items() if len(imgs) >= 4]
print(f"Found {len(valid_identities)} identities with at least 4 images.")

target_identities_count = 5000
# IMPORTANT: in v1: target_identities_count = 1000

if len(valid_identities) < target_identities_count:
    raise ValueError(f"Not enough identities with 4+ images to sample {target_identities_count}!")

# 3. Sample target_identities_count identities
sampled_identities = random.sample(valid_identities, target_identities_count)

print("Copying and renaming sampled images...")
copied_count = 0

# 4. Copy and rename images
for identity in sampled_identities:
    # Grab exactly 4 random images for this identity
    sampled_imgs = random.sample(identity_to_images[identity], 4)
    
    for count, orig_fname in enumerate(sampled_imgs, start=1):
        # Construct the new filename: {identity}_{count}.jpg
        new_fname = f"{identity}_{count}.jpg"
        
        src_path = os.path.join(img_dir, orig_fname)
        dst_path = os.path.join(output_imgs, new_fname)
        
        # Check if source exists to prevent crashes if the image folder is incomplete
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copied_count += 1
        else:
            print(f"Warning: {orig_fname} not found in {img_dir}")

    if copied_count % 400 == 0:
        print(f"  Processed {copied_count // 4} / {target_identities_count} identities...")

print(f"\nDone.")
print(f"  Total images copied: {copied_count}")
print(f"  Images  →  {output_imgs}/")