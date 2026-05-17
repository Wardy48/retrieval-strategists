"""
Celebrity Retrieval — Test Submission
======================================
Loads the best fine-tuned model (best_model.pth) and runs retrieval on
the test split, then submits to the grading server.

Usage:
    python submit_results.py
    python submit_results.py --url http://<server>/retrieval/  --group "your_group_name"
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch.nn as nn
import requests


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = False):
        super().__init__()
        backbone = resnet50(weights=None)
        self.backbone  = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        return self.embedding(self.backbone(x))


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


@torch.no_grad()
def extract_features(model, folder, batch_size=128):
    model.eval()
    filenames = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in filenames]

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([eval_transform(img) for img in images[i:i + batch_size]]).to(device)
        feats = model(batch)
        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0), filenames


def submit(results: dict, groupname: str, url: str):
    payload = json.dumps({"groupname": groupname, "images": results})
    response = requests.post(url, payload)
    try:
        result = json.loads(response.text)
        print(f"Server response — accuracy: {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"Server response (raw): {response.text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--group",      default="retrieval_strategists",
                        help="Your group name for submission")
    parser.add_argument("--url",        default="http://localhost:3001/retrieval/",
                        help="Grading server URL")
    parser.add_argument("--top_k",      type=int, default=10)
    args = parser.parse_args()

    # ── Paths ──
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    data_root     = os.path.join(script_dir,
                        "retrieval_strategists_full_dataset_in_partitions_v1")
    query_folder  = os.path.join(data_root, "test", "query")
    gallery_folder= os.path.join(data_root, "test", "gallery")
    ckpt_path     = os.path.join(script_dir, args.checkpoint)

    # ── Load model ──
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    embedding_dim = ckpt.get("embedding_dim", 512)
    print(f"  embedding_dim={embedding_dim}  "
          f"trained_epoch={ckpt.get('epoch', '?')}  "
          f"val_score={ckpt.get('val_score', '?'):.1f}/1000")

    model = EmbeddingModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Extract features ──
    print(f"\nExtracting query features   ({query_folder}) ...")
    query_feats,   query_files   = extract_features(model, query_folder)
    print(f"  {len(query_files)} query images")

    print(f"Extracting gallery features ({gallery_folder}) ...")
    gallery_feats, gallery_files = extract_features(model, gallery_folder)
    print(f"  {len(gallery_files)} gallery images")

    # ── Retrieve top-K ──
    print(f"\nComputing cosine similarity and retrieving top-{args.top_k} ...")
    sim = torch.matmul(query_feats, gallery_feats.T)
    _, top_k_idx = torch.topk(sim, k=args.top_k, dim=1)

    results = {}
    for i, qf in enumerate(query_files):
        results[qf] = [gallery_files[j] for j in top_k_idx[i].tolist()]

    # ── Preview ──
    print("\nSample predictions (first 3 queries):")
    for qf in query_files[:3]:
        print(f"  {qf}  →  {results[qf][:3]} ...")

    # ── Submit ──
    print(f"\nSubmitting to {args.url}  as group '{args.group}' ...")
    submit(results, groupname=args.group, url=args.url)


if __name__ == "__main__":
    main()
