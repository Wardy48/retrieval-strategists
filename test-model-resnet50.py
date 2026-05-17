import os
import json
from PIL import Image
import requests
import torch
from torchvision.models import resnet50, ResNet50_Weights

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def submit(results, groupname, url):
    res = {}
    res['groupname'] = groupname
    res['images'] = results
    res = json.dumps(res)
    # print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


def batching(images, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        tmp_images = images[i:i+batch_size]
        inputs = torch.stack([preprocess(img.convert("RGB")) for img in tmp_images]).to(device)
        with torch.no_grad():
            tmp_features = model(inputs)
            features.append(tmp_features)
    return torch.cat(features, dim=0)
        

data_folder = "/Users/prota/code/competition/test_data"
query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

query_images = []
query_filenames = []
gallery_images = []
gallery_filenames = []

for filename in os.listdir(query_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(query_folder, filename)
        query_filenames.append(filename)
        img = Image.open(img_path)
        query_images.append(img)

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(gallery_folder, filename)
        gallery_filenames.append(filename)
        img = Image.open(img_path)
        gallery_images.append(img)
        
# Print the number of images in each folder
print(f"Number of images in query folder: {len(query_images)}")
print(f"Number of images in gallery folder: {len(gallery_images)}")


##########
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()  # use penultimate-layer features for retrieval
model = model.to(device)
preprocess = weights.transforms()
model.eval()

print("Processing query images...")
query_features = batching(query_images, batch_size=16)
print("Processing gallery images...")
gallery_features = batching(gallery_images, batch_size=256)

# Normalize features
print("Normalizing features...")
query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)

print("Computing cosine similarity matrix...")
similarity_matrix = torch.matmul(query_features, gallery_features.T)

# Get top 10 matches for each query
print("Getting top 10 matches for each query...")
top_k = 10
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

# Convert indices to filenames
top_k_filenames = []
for i in range(top_k_indices.shape[0]):
    top_k_filenames.append([gallery_filenames[idx] for idx in top_k_indices[i]])
    
# Print the top 10 matches for each query
for i, query_filename in enumerate(query_filenames):
    print(f"Top {top_k} matches for {query_filename}:")
    for j, gallery_filename in enumerate(top_k_filenames[i]):
        print(f"  {j+1}: {gallery_filename}")
        
# Save the results to a dictionary
results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = top_k_filenames[i]
    
# Submit the results
submit(results=results, groupname="resnet50-imagenet", url="http://localhost:3001/retrieval/")