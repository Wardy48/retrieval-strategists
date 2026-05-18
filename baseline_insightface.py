import os
import sys
import json
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

from model_evaluator import evaluate_retrieval, score

#InsightFace baseline — extracts ArcFace pretrained embeddings.


# CONFIGURATION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

split = sys.argv[1] if len(sys.argv) > 1 else 'validation'
data_folder = os.path.join(DATA_DIR, split)
output_file = os.path.join(RESULTS_DIR, f'insightface_{split}.json')

print(f"Split: {split}")
print(f"Data folder: {data_folder}")


# LOAD MODEL

print("\nLoading InsightFace model...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
print("Model loaded\n")


# EXTRACTION FUNCTIONS

def extract_embedding(image_path):


    img = cv2.imread(image_path)

    if img is None:
        return None  # image failed to load

    faces = app.get(img)

    if len(faces) == 0:
        return None  # no face detected

    if len(faces) == 1:
        return faces[0].embedding

    largest_face = max(faces, key=lambda face: ((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))
    return largest_face.embedding  #numpy array of shape (512,) — the face embedding, None if no face detected



def extract_all(folder_path):
  


    filenames = []
    embeddings_list = []
    failure_count = 0

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        full_path = os.path.join(folder_path, filename)
        result = extract_embedding(full_path)

        if result is None:
            embedding = np.zeros(512, dtype=np.float32)
            failure_count += 1
        else:
            embedding = result

        embeddings_list.append(embedding)
        filenames.append(filename)

    print(f"Failed embeddings: {failure_count} out of {len(filenames)}")
    embeddings_array = np.stack(embeddings_list)

    return filenames, embeddings_array #filenames: list of filenames, embeddings are numpy array of shape (N, 512)


# EXTRACT FROM QUERY AND GALLERY

query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

print("Extracting query embeddings...")
query_filenames, query_embeddings = extract_all(query_folder)

print("Extracting gallery embeddings...")
gallery_filenames, gallery_embeddings = extract_all(gallery_folder)

print(f"\nQuery embeddings shape: {query_embeddings.shape}")
print(f"Gallery embeddings shape: {gallery_embeddings.shape}")


# NORMALIZE, SIMILARITY, TOP-K

query_features = torch.from_numpy(query_embeddings).float()
gallery_features = torch.from_numpy(gallery_embeddings).float()

query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)

similarity_matrix = torch.matmul(query_features, gallery_features.T)

_, top_k_indices = torch.topk(similarity_matrix, k=10, dim=1)


# BUILD RESULTS DICT AND SAVE

results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = [gallery_filenames[idx] for idx in top_k_indices[i].tolist()]

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")


# EVALUATE
metrics = evaluate_retrieval(results)
total_score = score(metrics)

print(f"\n--- InsightFace Results on {split} split ---")
print(f"Top-1:  {metrics['top_1']:.4f}")
print(f"Top-5:  {metrics['top_5']:.4f}")
print(f"Top-10: {metrics['top_10']:.4f}")
print(f"Total score: {total_score:.1f} / 1000")
print(f"Queries evaluated: {metrics['no_of_queries']}")

