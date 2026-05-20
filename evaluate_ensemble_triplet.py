import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Import your colleague's evaluation functions
# (Assuming your colleague's code is saved as colleague_eval.py)
# from colleague_eval import evaluate_retrieval, score

# ==========================================================
# 1. RE-DECLARE YOUR MODEL ARCHITECTURE (From your training script)
# ==========================================================
class ModifiedCNN(nn.Module):
    def __init__(self, base_features, in_channels, embedding_dim=256):
        super(ModifiedCNN, self).__init__()
        self.base_features = base_features
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, 128, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * 3, embedding_dim)
        
    def forward(self, x):
        x = self.base_features(x)
        c1 = self.pool(self.relu(self.conv1(x))).view(x.size(0), -1)
        c2 = self.pool(self.relu(self.conv2(x))).view(x.size(0), -1)
        c3 = self.pool(self.relu(self.conv3(x))).view(x.size(0), -1)
        return self.fc(torch.cat((c1, c2, c3), dim=1))

def load_ensemble(checkpoint_dir="/home/disi/retrivial_strategist/ensemble_checkpoints", device="cuda"):
    # 1. MobileNetV2
    mobilenet = models.mobilenet_v2()
    model_mn = ModifiedCNN(mobilenet.features, in_channels=1280, embedding_dim=256)
    model_mn.load_state_dict(torch.load(f"{checkpoint_dir}/mod_mobilenet.pth", map_location=device))
    
    # 2. DenseNet201
    densenet = models.densenet201(memory_efficient=True)
    model_dn = ModifiedCNN(densenet.features, in_channels=1920, embedding_dim=256)
    model_dn.load_state_dict(torch.load(f"{checkpoint_dir}/mod_densenet.pth", map_location=device))
    
    # 3. Vision Transformer (ViT)
    model_vit = models.vit_b_16()
    model_vit.heads.head = nn.Linear(model_vit.heads.head.in_features, 256)
    model_vit.load_state_dict(torch.load(f"{checkpoint_dir}/vit.pth", map_location=device))
    
    model_mn.to(device).eval()
    model_dn.to(device).eval()
    model_vit.to(device).eval()
    
    return model_mn, model_dn, model_vit

# ==========================================================
# 2. FEATURE EXTRACTION PIPELINE
# ==========================================================


def extract_ensemble_features(image_paths, models_ensemble, transform, device):
    model_mn, model_dn, model_vit = models_ensemble
    
    # ==========================================================
    # IL FIX: Spegni Dropout e blocca le statistiche di BatchNorm!
    # ==========================================================
    model_mn.eval()
    model_dn.eval()
    model_vit.eval()
    
    features_dict = {}
    
    print(f"[*] Extracting features for {len(image_paths)} images...")
    with torch.no_grad():
        for path in image_paths:
            filename = os.path.basename(path)
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract individual embeddings
            emb_mn = model_mn(tensor)
            emb_dn = model_dn(tensor)
            emb_vit = model_vit(tensor)
            
            # CRITICAL ENSEMBLE STEP: L2-Normalize each model's output BEFORE concatenating
            emb_mn = F.normalize(emb_mn, p=2, dim=1)
            emb_dn = F.normalize(emb_dn, p=2, dim=1)
            emb_vit = F.normalize(emb_vit, p=2, dim=1)
            
            # Concatenate into a unified 768-dimensional descriptor (256 * 3)
            combined_emb = torch.cat((emb_mn, emb_dn, emb_vit), dim=1)
            features_dict[filename] = combined_emb.cpu().numpy().flatten()
            
    return features_dict

# ==========================================================
# 3. BRIDGING LOOP (GENERATE TOP-10 RESULTS)
# ==========================================================
def build_results_dict(query_features, gallery_features):
    results = {}
    
    gallery_filenames = list(gallery_features.keys())
    # Convert gallery features map to a single large numpy array for fast matrix operations
    gallery_matrix = np.array([gallery_features[f] for f in gallery_filenames])
    
    print("[*] Computing distances and ranking gallery images...")
    for q_filename, q_emb in query_features.items():
        # Compute Euclidean Distance from this query to all gallery vectors
        distances = np.linalg.norm(gallery_matrix - q_emb, axis=1)
        
        # Get indices of the 10 lowest distances
        top_10_indices = np.argsort(distances)[:10]
        
        # Map indices back to filenames
        results[q_filename] = [gallery_filenames[idx] for idx in top_10_indices]
        
    return results

# ==========================================================
# COLLEAGUE'S ORIGINAL CODE (Pasted here for self-containment)
# ==========================================================
def get_identity(filename):
    """
    Extracts the identity number from a filename using regex.
    Assumes the filename contains a numeric ID (e.g., '12345.jpg' -> 12345).
    """
    match = re.search(r'\d+', os.path.basename(filename))
    if match:
        return int(match.group(0))
    return None

def evaluate_retrieval(results, identity_func=get_identity):
    top_1, top_5, top_10, total_queries = 0, 0, 0, 0
    for query_filename, gallery_list in results.items():
        query_id = identity_func(query_filename)
        if query_id is None:
            continue
        total_queries += 1
        gallery_ids = [identity_func(f) for f in gallery_list[:10]]
        if gallery_ids[0] == query_id: top_1 += 1
        if query_id in gallery_ids[:5]: top_5 += 1
        if query_id in gallery_ids[:10]: top_10 += 1
        
    if total_queries == 0:
        return {'top_1': 0, 'top_5': 0, 'top_10': 0, 'no_of_queries': 0}
    return {'top_1': top_1 / total_queries, 'top_5': top_5 / total_queries, 'top_10': top_10 / total_queries, 'no_of_queries': total_queries}

def score(metrics):
    return (metrics['top_1'] * 600 + metrics['top_5'] * 300 + metrics['top_10'] * 100)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
# ==========================================================
# FIXED MAIN EXECUTION BLOCK
# ==========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define validation evaluation transforms (Matches training resolution)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # FIXED PATHS: Included the 'val' split folder
    QUERY_DIR = "/home/disi/retrivial_strategist/dataset_final_flat/val/query" 
    GALLERY_DIR = "/home/disi/retrivial_strategist/dataset_final_flat/val/gallery"
    
    # Catch any variation of image extension extensions securely
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    
    query_paths = []
    for ext in valid_extensions:
        query_paths.extend(glob.glob(os.path.join(QUERY_DIR, ext)))
        
    gallery_paths = []
    for ext in valid_extensions:
        gallery_paths.extend(glob.glob(os.path.join(GALLERY_DIR, ext)))
        
    print(f"[*] Found {len(query_paths)} query images.")
    print(f"[*] Found {len(gallery_paths)} gallery images.")
    # ==========================================================
    # DATASET SANITY CHECK: Esistono i match?
    # ==========================================================
    query_ids = set([get_identity(f) for f in query_paths if get_identity(f) is not None])
    gallery_ids = set([get_identity(f) for f in gallery_paths if get_identity(f) is not None])
    
    overlap = query_ids.intersection(gallery_ids)
    
    print("\n--- DATASET SANITY CHECK ---")
    print(f"ID Unici nelle Queries: {len(query_ids)}")
    print(f"ID Unici nella Gallery: {len(gallery_ids)}")
    print(f"ID in comune (Possibili Match): {len(overlap)}")
    print("----------------------------\n")
    
    if len(overlap) == 0:
        print("[!] ALLARME ROSSO: Nessun ID della query esiste nella gallery!")
        print("[!] L'accuratezza massima teorica è 0.00%.")
    
    if len(query_paths) == 0 or len(gallery_paths) == 0:
        print("[!] ERROR: No images found. Check your absolute paths or permissions.")
        exit(1)
    
    # Load your trained models using the absolute defaults we fixed earlier
    ensemble = load_ensemble(device=device)
    
    # Extract features
    query_features = extract_ensemble_features(query_paths, ensemble, eval_transform, device)
    gallery_features = extract_ensemble_features(gallery_paths, ensemble, eval_transform, device)
    
    # Build the required dictionary for your colleague's engine
    calculated_results = build_results_dict(query_features, gallery_features)
    
    # Run colleague's scoring script
    metrics = evaluate_retrieval(calculated_results)
    total_score = score(metrics)

    # Build the required dictionary for your colleague's engine
    calculated_results = build_results_dict(query_features, gallery_features)
    
    # ==========================================================
    # BLOCCO DIAGNOSTICO: Vediamo cosa vede davvero lo script!
    # ==========================================================
    print("\n--- DIAGNOSTIC CHECK ---")
    sample_q = list(calculated_results.keys())[0]
    sample_g_list = calculated_results[sample_q][:3]
    
    print(f"Query: {sample_q}  -> ID Estratto: {get_identity(sample_q)}")
    for g in sample_g_list:
        print(f"Gallery Top Match: {g} -> ID Estratto: {get_identity(g)}")
    print("------------------------")
    
    # ==========================================================
    # FIX CRITICO: Forza Python a usare la NOSTRA get_identity
    # ==========================================================
    metrics = evaluate_retrieval(calculated_results, identity_func=get_identity)
    
    total_score = score(metrics)
    
    print(f"\n==========================================")
    print(f"       ENSEMBLE RETRIEVAL REPORT")
    print(f"==========================================")
    print(f"Evaluated Queries: {metrics['no_of_queries']}")
    print(f"Top-1 Accuracy:    {metrics['top_1']*100:.2f}%")
    print(f"Top-5 Accuracy:    {metrics['top_5']*100:.2f}%")
    print(f"Top-10 Accuracy:   {metrics['top_10']*100:.2f}%")
    print(f"------------------------------------------")
    print(f"FINAL COMPETITION SCORE: {total_score:.1f} / 1000")
    print(f"==========================================")