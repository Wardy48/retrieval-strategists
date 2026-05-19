"""
CLIP Zero-Shot Retrieval
========================
Evaluates CLIP without any fine-tuning — no training required.
Uses CLIP's pre-trained weights straight out of the box to extract
embeddings and rank gallery images by cosine similarity to each query.

This is a zero-shot baseline: no gradient updates, no ArcFace loss,
just pure CLIP embeddings. Useful for comparing against fine-tuned models.

Usage:
    pip install open-clip-torch
    python evaluate_clip_zeroshot.py
"""

import os
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _get_device()
print(f"Using device: {device}")

_BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT   = os.path.join(_BASE, "dataset_final")
VAL_CELEB   = os.path.join(DATA_ROOT, "val", "celebrities")

_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])


@torch.no_grad()
def extract_features(model, folder, batch_size=64):
    model.eval()
    file_paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_paths.append(os.path.join(root, f))
    file_paths.sort()

    images = [Image.open(p).convert("RGB") for p in file_paths]
    all_feats = []
    for i in tqdm(range(0, len(images), batch_size), desc="Extracting"):
        batch = torch.stack([eval_transform(img) for img in images[i:i+batch_size]]).to(device)
        feats = model.encode_image(batch)
        feats = F.normalize(feats.float(), p=2, dim=1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0), file_paths


def evaluate(model, val_celeb_folder, top_k=10):
    print(f"\nExtracting features from {val_celeb_folder} ...")
    all_feats, all_paths = extract_features(model, val_celeb_folder)
    all_ids = [os.path.basename(os.path.dirname(p)) for p in all_paths]

    id_to_indices = defaultdict(list)
    for i, id_ in enumerate(all_ids):
        id_to_indices[id_].append(i)

    query_indices = [indices[0] for indices in id_to_indices.values()]
    query_feats   = all_feats[query_indices]
    query_ids     = [all_ids[i] for i in query_indices]

    sim = torch.matmul(query_feats, all_feats.T)
    for qi, gi in enumerate(query_indices):
        sim[qi, gi] = -2.0

    _, top_k_idx = torch.topk(sim, k=top_k, dim=1)

    top1 = top5 = top10 = 0
    n = len(query_indices)
    for i, q_id in enumerate(query_ids):
        retrieved = [all_ids[j] for j in top_k_idx[i].tolist()]
        if retrieved[0] == q_id:     top1  += 1
        if q_id in retrieved[:5]:    top5  += 1
        if q_id in retrieved[:10]:   top10 += 1

    score = (top1 / n) * 600 + (top5 / n) * 300 + (top10 / n) * 100
    print(f"\nZero-shot CLIP results:")
    print(f"  Top-1:  {top1/n:.3f}  ({top1}/{n})")
    print(f"  Top-5:  {top5/n:.3f}  ({top5}/{n})")
    print(f"  Top-10: {top10/n:.3f}  ({top10}/{n})")
    print(f"  Score:  {score:.1f}/1000")
    return score


if __name__ == "__main__":
    print("\nLoading CLIP ViT-B/32 (openai) ...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()

    evaluate(model, VAL_CELEB)
