import os
from datasets import load_dataset

# 1. Define your dataset and where to save it
dataset_name = "bitmind/SyntheticFacesHQ"
save_folder = "Synthetic_Faces_Subset"     # Folder to save the images
target_image_count = 5000                  # How many images you want
# IMPORTANT: in v1: target_image_count = 1000

# Create the folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

print(f"Connecting to {dataset_name}...")

# 2. STREAM the dataset (This avoids downloading the massive 80GB files)
dataset = load_dataset(dataset_name, split="train", streaming=True)

print(f"Downloading {target_image_count} images to '{save_folder}'...")

# 3. Loop through the stream and save exactly what you need
for i, item in enumerate(dataset):
    if i >= target_image_count:
        break
    
    # Extract the image. (In most Hugging Face vision datasets, the column is called 'image')
    img = item["image"]
    
    # Save it locally
    file_path = os.path.join(save_folder, f"image_{i}.jpg")
    img.save(file_path)
    
    # Print a progress update every 100 images
    if (i + 1) % 100 == 0:
        print(f"Saved {i + 1} / {target_image_count} images...")

print("Done! You successfully grabbed a subset without filling up your hard drive.")