import os
import random
import shutil
import zipfile

zip_path = "Img/Cartoon.zip"
extract_dir = "Cartoon_all"
output_dir = "Cartoon_Faces_Subset"
target_image_count = 5000
# IMPORTANT: in v1: target_image_count = 1000

os.makedirs(extract_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- Unzip (skip if already done) ---
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

# --- Find all images recursively ---
print("Scanning for images...")
all_images = []
for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(root, fname))

print(f"Found {len(all_images)} images total.")

if len(all_images) < target_image_count:
    raise ValueError(f"Only found {len(all_images)} images, need {target_image_count}.")

# --- Sample and copy ---
sampled = random.sample(all_images, target_image_count)

for i, src_path in enumerate(sampled):
    # Extract the original file extension (e.g., '.jpg', '.png')
    ext = os.path.splitext(src_path)[1] 
    
    # Create a guaranteed unique filename using the loop index
    new_fname = f"c{i:04d}{ext}" 
    
    dst_path = os.path.join(output_dir, new_fname)
    shutil.copy(src_path, dst_path)
    
    if (i + 1) % 100 == 0:
        print(f"Copied {i + 1} / {target_image_count}...")

print(f"Done. {target_image_count} cartoon images saved to {output_dir}/")