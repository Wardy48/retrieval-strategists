"""
Celebrity Retrieval — CLIP + ArcFace Fine-tuning
=================================================
Replaces the ResNet50 backbone with OpenAI's CLIP vision encoder.
CLIP was pre-trained on 400M image-text pairs across wildly different
visual domains, making it far more robust to the real→synthetic domain
gap in this challenge.

Architecture:
    CLIP ViT-B/32 vision encoder → 512-dim projection → ArcFace loss

Usage:
    pip install torch torchvision pillow tqdm open-clip-torch
    python train_clip.py

Expected score improvement: ~400 (ResNet50) → 600-800+ (CLIP)
"""

import os
import math
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import open_clip


# ─── Device ───────────────────────────────────────────────────────────────────
def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _get_device()
if __name__ == "__main__":
    print(f"Using device: {device}")


# ─── Paths ────────────────────────────────────────────────────────────────────
_BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT     = os.path.join(_BASE, "dataset_final")
TRAIN_CELEB   = os.path.join(DATA_ROOT, "train", "celebrities")
VAL_CELEB     = os.path.join(DATA_ROOT, "val",   "celebrities")
SAVE_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_clip_model.pth")


# ─── Hyper-parameters ─────────────────────────────────────────────────────────
CLIP_MODEL    = "ViT-B-32"          # CLIP backbone variant
CLIP_PRETRAIN = "openai"            # pretrained weights
EMBEDDING_DIM = 512                 # CLIP ViT-B/32 output dim
BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
ARC_S         = 64.0
ARC_M         = 0.50


# ─── ArcFace Loss ─────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings, p=2), F.normalize(self.weight, p=2))
        sine   = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return F.cross_entropy(output, labels)


# ─── CLIP Embedding Model ─────────────────────────────────────────────────────
class CLIPEmbeddingModel(nn.Module):
    """
    CLIP vision encoder with a lightweight projection head.
    The CLIP backbone is partially frozen — only the last 4 transformer
    blocks are fine-tuned to preserve general visual knowledge while
    adapting to celebrity identity matching.
    """
    def __init__(self, clip_model_name: str = "ViT-B-32",
                 pretrained: str = "openai", embedding_dim: int = 512):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.visual = clip_model.visual

        # Freeze all layers first
        for param in self.visual.parameters():
            param.requires_grad = False

        # Unfreeze the last 4 transformer blocks for fine-tuning
        if hasattr(self.visual, 'transformer'):
            blocks = list(self.visual.transformer.resblocks)
            for block in blocks[-4:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze the final projection
        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            self.visual.proj.requires_grad = True

        clip_out_dim = self.visual.output_dim
        self.projection = nn.Sequential(
            nn.Linear(clip_out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x)           # (B, clip_out_dim)
        if features.dim() > 2:
            features = features[:, 0]       # take CLS token if needed
        return self.projection(features)    # (B, embedding_dim)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CelebrityDataset(Dataset):
    """
    Loads celebrity images from per-identity subdirectories.
    Identity label = directory name.
    """
    def __init__(self, celeb_folder: str, transform=None):
        self.transform = transform
        raw = []
        all_identities = set()

        for identity_dir in sorted(os.listdir(celeb_folder)):
            id_path = os.path.join(celeb_folder, identity_dir)
            if not os.path.isdir(id_path):
                continue
            for fname in sorted(os.listdir(id_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_identities.add(identity_dir)
                    raw.append((os.path.join(id_path, fname), identity_dir))

        self.identity_to_idx = {
            ident: idx for idx, ident in enumerate(sorted(all_identities))
        }
        self.num_classes = len(self.identity_to_idx)
        self.samples = [(path, self.identity_to_idx[ident]) for path, ident in raw]
        print(f"  Loaded {len(self.samples):,} images | {self.num_classes:,} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Transforms ───────────────────────────────────────────────────────────────
# CLIP expects 224×224 with specific normalisation
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])


# ─── Feature Extraction ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model: nn.Module, folder: str,
                     transform, batch_size: int = 64):
    model.eval()
    file_paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_paths.append(os.path.join(root, f))
    file_paths.sort()

    images = [Image.open(p).convert("RGB") for p in file_paths]
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([transform(img) for img in images[i:i + batch_size]]).to(device)
        feats = model(batch)
        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0), file_paths


# ─── Validation ───────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, val_celeb_folder: str,
             transform, top_k: int = 10) -> float:
    all_feats, all_paths = extract_features(model, val_celeb_folder, transform)
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
    print(f"    Top-1: {top1/n:.3f} | Top-5: {top5/n:.3f} | "
          f"Top-10: {top10/n:.3f} | Score: {score:.1f}/1000")
    return score


# ─── Main Training Loop ───────────────────────────────────────────────────────
def main():
    print(f"\n── CLIP Model: {CLIP_MODEL} ({CLIP_PRETRAIN}) ────────────────────")
    print(f"   Train: {TRAIN_CELEB}")
    print(f"   Val:   {VAL_CELEB}")

    print("\n── Building dataset ─────────────────────────────────────")
    train_dataset = CelebrityDataset(TRAIN_CELEB, transform=train_transform)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if device.type == "mps" else 4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print("\n── Building model ───────────────────────────────────────")
    model = CLIPEmbeddingModel(
        clip_model_name=CLIP_MODEL,
        pretrained=CLIP_PRETRAIN,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    arcface = ArcFaceLoss(
        embedding_dim=EMBEDDING_DIM,
        num_classes=train_dataset.num_classes,
        s=ARC_S,
        m=ARC_M,
    ).to(device)

    # Separate LRs: tiny for frozen backbone layers that are unfrozen,
    # full LR for the new projection head and ArcFace
    trainable_backbone = [p for p in model.visual.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone,           "lr": LR * 0.1},
        {"params": model.projection.parameters(), "lr": LR},
        {"params": arcface.parameters(),          "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    warmup_epochs = 2

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n── Training: {EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"emb={EMBEDDING_DIM}, ArcFace s={ARC_S} m={ARC_M} ────")

    best_score = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings = model(imgs)
            loss = arcface(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arcface.parameters()), max_norm=5.0
            )
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        lr_now   = optimizer.param_groups[1]["lr"]
        print(f"\nEpoch [{epoch+1:02d}/{EPOCHS}] loss={avg_loss:.4f}  lr={lr_now:.2e}")

        score = evaluate(model, VAL_CELEB, eval_transform)

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch":         epoch + 1,
                    "model_state":   model.state_dict(),
                    "embedding_dim": EMBEDDING_DIM,
                    "clip_model":    CLIP_MODEL,
                    "val_score":     score,
                },
                SAVE_PATH,
            )
            print(f"    ✓ New best — saved to {SAVE_PATH}")

        scheduler.step()

    print(f"\n═══════════════════════════════════════════════════")
    print(f"Training complete.  Best validation score: {best_score:.1f}/1000")
    print(f"Best checkpoint saved at: {SAVE_PATH}")
    print(f"═══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
