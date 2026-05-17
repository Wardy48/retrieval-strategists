import os
import math
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _get_device()
if __name__ == "__main__":
    print(f"Using device: {device}")

DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "retrieval_strategists_full_dataset_in_partitions_v1"
)
TRAIN_QUERY   = os.path.join(DATA_ROOT, "training",   "query")
TRAIN_GALLERY = os.path.join(DATA_ROOT, "training",   "gallery")
VAL_QUERY     = os.path.join(DATA_ROOT, "validation", "query")
VAL_GALLERY   = os.path.join(DATA_ROOT, "validation", "gallery")
SAVE_PATH     = os.path.join(os.path.dirname(__file__), "best_model.pth")

EMBEDDING_DIM = 512
BATCH_SIZE    = 64
EPOCHS        = 25
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
ARC_S         = 64.0
ARC_M         = 0.50


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Pushes same-identity embeddings together in angular space while
    separating different identities — ideal for cross-domain retrieval.

    Reference: Deng et al., ArcFace: Additive Angular Margin Loss for
    Deep Face Recognition, CVPR 2019.
    """
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

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return F.cross_entropy(output, labels)


class EmbeddingModel(nn.Module):

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features   = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings


class CelebrityDataset(Dataset):
    """
    Loads images from one or more flat folders.
    Identity labels are parsed from filenames: {celebrity_id}_{image_num}.jpg
    """
    def __init__(self, folders: list, transform=None):
        self.transform = transform
        raw = []
        all_identities: set = set()

        for folder in folders:
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    identity = fname.rsplit('_', 1)[0]  # e.g. "10001"
                    all_identities.add(identity)
                    raw.append((os.path.join(folder, fname), identity))

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


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),   # makes features colour-agnostic
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


@torch.no_grad()
def extract_features(model: nn.Module, folder: str,
                     transform, batch_size: int = 128):
    """Returns (N, D) normalised feature tensor and sorted filenames."""
    model.eval()
    filenames = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in filenames]

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([transform(img) for img in images[i:i + batch_size]]).to(device)
        feats = model(batch)
        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0), filenames


def evaluate(model: nn.Module, query_folder: str, gallery_folder: str,
             transform, top_k: int = 10) -> float:
    """
    Computes Top-1 / Top-5 / Top-10 retrieval accuracy and the
    competition score (out of 1000).
    """
    query_feats,   query_files   = extract_features(model, query_folder,   transform)
    gallery_feats, gallery_files = extract_features(model, gallery_folder, transform)

    query_ids   = [f.rsplit('_', 1)[0] for f in query_files]
    gallery_ids = [f.rsplit('_', 1)[0] for f in gallery_files]

    sim = torch.matmul(query_feats, gallery_feats.T)
    _, top_k_idx = torch.topk(sim, k=top_k, dim=1)

    top1 = top5 = top10 = 0
    n = len(query_files)
    for i, q_id in enumerate(query_ids):
        retrieved = [gallery_ids[j] for j in top_k_idx[i].tolist()]
        if retrieved[0] == q_id:        top1  += 1
        if q_id in retrieved[:5]:       top5  += 1
        if q_id in retrieved[:10]:      top10 += 1

    score = (top1 / n) * 600 + (top5 / n) * 300 + (top10 / n) * 100
    print(f"    Top-1: {top1/n:.3f} | Top-5: {top5/n:.3f} | "
          f"Top-10: {top10/n:.3f} | Score: {score:.1f}/1000")
    return score


def main():
    print("\n── Building dataset ─────────────────────────────────────")
    train_dataset = CelebrityDataset(
        folders=[TRAIN_QUERY, TRAIN_GALLERY],
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if device.type == "mps" else 4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print("\n── Building model ───────────────────────────────────────")
    model   = EmbeddingModel(embedding_dim=EMBEDDING_DIM).to(device)
    arcface = ArcFaceLoss(
        embedding_dim=EMBEDDING_DIM,
        num_classes=train_dataset.num_classes,
        s=ARC_S,
        m=ARC_M,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),  "lr": LR * 0.1},
        {"params": model.embedding.parameters(), "lr": LR},
        {"params": arcface.parameters(),         "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    warmup_epochs = 3

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
        # ── train ──
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
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

        avg_loss = total_loss / len(train_loader)
        lr_now   = optimizer.param_groups[1]["lr"]
        print(f"\nEpoch [{epoch+1:02d}/{EPOCHS}] loss={avg_loss:.4f}  lr={lr_now:.2e}")

        # ── validate ──
        score = evaluate(model, VAL_QUERY, VAL_GALLERY, eval_transform)

        # ── checkpoint ──
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch":          epoch + 1,
                    "model_state":    model.state_dict(),
                    "embedding_dim":  EMBEDDING_DIM,
                    "val_score":      score,
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
