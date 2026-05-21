"""
Multi-model comparison on retrieval_strategists_full_dataset_in_partitions_v3.

Runs three models on the validation split (or test via --split test):
  1. InsightFace buffalo_l  (pretrained ArcFace, no training needed)
  2. CLIP ViT-B/32          (zero-shot image embeddings)
  3. Custom ResNet50+ArcFace (best_model.pth, trained on dataset_final)

Scoring: Top-1×600 + Top-5×300 + Top-10×100  (max 1000)

Usage:
    python compare_models_v3.py              # validation split
    python compare_models_v3.py --split test
    python compare_models_v3.py --skip insightface,clip   # run only custom
    python compare_models_v3.py --save-json  # also write results/*.json per model
    python compare_models_v3.py --visualize          # save retrieval grid (5 random queries)
    python compare_models_v3.py --visualize --n-vis 10 --top-vis 5  # 10 queries, top-5 shown
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, 'retrieval_strategists_full_dataset_in_partitions_v3')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'best_model.pth')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--split',     default='validation', choices=['validation', 'test'])
parser.add_argument('--skip',      default='',  help='comma-separated models to skip: insightface,clip,custom')
parser.add_argument('--save-json', action='store_true', help='save per-model results JSON')
parser.add_argument('--visualize', action='store_true', help='save retrieval grid image to results/')
parser.add_argument('--n-vis',     type=int, default=5,  help='number of query rows to show (default 5)')
parser.add_argument('--top-vis',   type=int, default=5,  help='number of top retrievals to show per query (default 5)')
args = parser.parse_args()

split    = args.split
skip_set = set(s.strip().lower() for s in args.skip.split(',') if s.strip())

QUERY_DIR   = os.path.join(DATA_ROOT, split, 'query')
GALLERY_DIR = os.path.join(DATA_ROOT, split, 'gallery')

print(f"\n{'='*60}")
print(f"  Multi-model comparison — {split.upper()} split")
print(f"  Dataset: retrieval_strategists_full_dataset_in_partitions_v3")
print(f"  Query:   {QUERY_DIR}")
print(f"  Gallery: {GALLERY_DIR}")
print(f"{'='*60}\n")


# ── device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():    return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print(f"Device: {device}\n")


# ── identity helper ───────────────────────────────────────────────────────────
def get_identity(filename):
    """Parse identity from filename like '10021_3.jpg' → '10021'."""
    return os.path.basename(filename).rsplit('_', 1)[0]


# ── scoring ───────────────────────────────────────────────────────────────────
def compute_score(results: dict) -> dict:
    """
    results: {query_filename: [gallery_filename, ...]}  (top-10 ordered list)
    Returns dict with top1/top5/top10 rates and final score.
    """
    top1 = top5 = top10 = 0
    n = len(results)
    for q_fname, gallery_list in results.items():
        q_id = get_identity(q_fname)
        retrieved_ids = [get_identity(g) for g in gallery_list[:10]]
        if retrieved_ids[0] == q_id:        top1  += 1
        if q_id in retrieved_ids[:5]:       top5  += 1
        if q_id in retrieved_ids[:10]:      top10 += 1
    return {
        'top1':  top1 / n,
        'top5':  top5 / n,
        'top10': top10 / n,
        'score': (top1/n)*600 + (top5/n)*300 + (top10/n)*100,
        'n':     n,
    }


def print_scores(name, metrics):
    print(f"  {'Top-1':>6}: {metrics['top1']*100:5.1f}%")
    print(f"  {'Top-5':>6}: {metrics['top5']*100:5.1f}%")
    print(f"  {'Top-10':>6}: {metrics['top10']*100:5.1f}%")
    print(f"  {'Score':>6}: {metrics['score']:.1f} / 1000\n")


# ── load image file lists ─────────────────────────────────────────────────────
def load_image_list(folder):
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    return files, [os.path.join(folder, f) for f in files]


q_fnames,  q_paths  = load_image_list(QUERY_DIR)
g_fnames,  g_paths  = load_image_list(GALLERY_DIR)
print(f"Queries:  {len(q_fnames)}  |  Gallery: {len(g_fnames)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — InsightFace buffalo_l
# ─────────────────────────────────────────────────────────────────────────────
insightface_results = None
if 'insightface' not in skip_set:
    print(f"{'─'*60}")
    print("  [1/3]  InsightFace buffalo_l  (pretrained ArcFace)")
    print(f"{'─'*60}")
    try:
        import cv2
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0)

        def extract_insightface(image_path):
            img = cv2.imread(image_path)
            if img is None:
                return None
            faces = app.get(img)
            if not faces:
                return None
            if len(faces) == 1:
                return faces[0].embedding
            return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding

        def extract_batch_insightface(paths, label):
            embs, fail = [], 0
            for p in tqdm(paths, desc=f'  InsightFace {label}'):
                e = extract_insightface(p)
                if e is None:
                    e = np.zeros(512, dtype=np.float32)
                    fail += 1
                embs.append(e)
            print(f"  {label}: {len(paths)} images, {fail} failed")
            return np.stack(embs)

        q_embs = extract_batch_insightface(q_paths, 'Query')
        g_embs = extract_batch_insightface(g_paths, 'Gallery')

        q_feat = F.normalize(torch.from_numpy(q_embs).float(), p=2, dim=1)
        g_feat = F.normalize(torch.from_numpy(g_embs).float(), p=2, dim=1)
        sim    = torch.matmul(q_feat, g_feat.T)
        _, top_idx = torch.topk(sim, k=10, dim=1)

        insightface_results = {
            q_fnames[i]: [g_fnames[j] for j in top_idx[i].tolist()]
            for i in range(len(q_fnames))
        }
        metrics = compute_score(insightface_results)
        print_scores('InsightFace', metrics)

        if args.save_json:
            out = os.path.join(RESULTS_DIR, f'insightface_v3_{split}.json')
            with open(out, 'w') as f: json.dump(insightface_results, f, indent=2)
            print(f"  Saved → {out}")

    except Exception as e:
        print(f"  InsightFace FAILED: {e}\n")
        metrics = None

    insightface_metrics = metrics if 'metrics' in dir() else None
else:
    print("  [1/3]  InsightFace — SKIPPED\n")
    insightface_metrics = None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — CLIP ViT-B/32  (zero-shot)
# ─────────────────────────────────────────────────────────────────────────────
clip_results = None
if 'clip' not in skip_set:
    print(f"{'─'*60}")
    print("  [2/3]  CLIP ViT-B/32  (zero-shot)")
    print(f"{'─'*60}")
    try:
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        clip_model = clip_model.to(device).eval()

        _CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
        _CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
        clip_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])

        @torch.no_grad()
        def extract_clip(paths, label, batch_size=64):
            all_feats = []
            for i in tqdm(range(0, len(paths), batch_size), desc=f'  CLIP {label}'):
                batch_paths = paths[i:i+batch_size]
                imgs = [Image.open(p).convert('RGB') for p in batch_paths]
                batch = torch.stack([clip_transform(img) for img in imgs]).to(device)
                feats = clip_model.encode_image(batch)
                feats = F.normalize(feats.float(), p=2, dim=1)
                all_feats.append(feats.cpu())
            return torch.cat(all_feats, dim=0)

        q_feat_clip = extract_clip(q_paths, 'Query')
        g_feat_clip = extract_clip(g_paths, 'Gallery')
        sim_clip    = torch.matmul(q_feat_clip, g_feat_clip.T)
        _, top_idx  = torch.topk(sim_clip, k=10, dim=1)

        clip_results = {
            q_fnames[i]: [g_fnames[j] for j in top_idx[i].tolist()]
            for i in range(len(q_fnames))
        }
        clip_metrics = compute_score(clip_results)
        print_scores('CLIP', clip_metrics)

        if args.save_json:
            out = os.path.join(RESULTS_DIR, f'clip_v3_{split}.json')
            with open(out, 'w') as f: json.dump(clip_results, f, indent=2)
            print(f"  Saved → {out}")

    except Exception as e:
        print(f"  CLIP FAILED: {e}\n")
        clip_metrics = None
else:
    print("  [2/3]  CLIP — SKIPPED\n")
    clip_metrics = None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Custom ResNet50 + ArcFace  (best_model.pth)
# ─────────────────────────────────────────────────────────────────────────────
custom_results = None
if 'custom' not in skip_set:
    print(f"{'─'*60}")
    print("  [3/3]  Custom ResNet50 + ArcFace  (best_model.pth)")
    print(f"{'─'*60}")
    try:
        # ── model definition (must match train_arcface.py exactly) ────────────
        class EmbeddingModel(nn.Module):
            def __init__(self, embedding_dim=512, pretrained=False):
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

        ckpt = torch.load(MODEL_PATH, map_location=device)
        emb_dim = ckpt.get('embedding_dim', 512)
        model   = EmbeddingModel(embedding_dim=emb_dim)
        model.load_state_dict(ckpt['model_state'])
        model = model.to(device).eval()

        trained_on = f"dataset_v{ckpt.get('dataset_version','?')}"
        val_score  = ckpt.get('val_score', '?')
        epoch      = ckpt.get('epoch', '?')
        print(f"  Loaded checkpoint: epoch={epoch}, trained_on={trained_on}, saved_val_score={val_score:.1f}")

        _MEAN = [0.485, 0.456, 0.406]
        _STD  = [0.229, 0.224, 0.225]
        custom_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

        @torch.no_grad()
        def extract_custom(paths, label, batch_size=128):
            all_feats = []
            for i in tqdm(range(0, len(paths), batch_size), desc=f'  Custom {label}'):
                batch_paths = paths[i:i+batch_size]
                imgs  = [Image.open(p).convert('RGB') for p in batch_paths]
                batch = torch.stack([custom_transform(img) for img in imgs]).to(device)
                feats = model(batch)
                feats = F.normalize(feats, p=2, dim=1)
                all_feats.append(feats.cpu())
            return torch.cat(all_feats, dim=0)

        q_feat_custom = extract_custom(q_paths, 'Query')
        g_feat_custom = extract_custom(g_paths, 'Gallery')
        sim_custom    = torch.matmul(q_feat_custom, g_feat_custom.T)
        _, top_idx    = torch.topk(sim_custom, k=10, dim=1)

        custom_results = {
            q_fnames[i]: [g_fnames[j] for j in top_idx[i].tolist()]
            for i in range(len(q_fnames))
        }
        custom_metrics = compute_score(custom_results)
        print_scores('Custom', custom_metrics)

        if args.save_json:
            out = os.path.join(RESULTS_DIR, f'custom_v3_{split}.json')
            with open(out, 'w') as f: json.dump(custom_results, f, indent=2)
            print(f"  Saved → {out}")

    except Exception as e:
        print(f"  Custom model FAILED: {e}\n")
        custom_metrics = None
else:
    print("  [3/3]  Custom model — SKIPPED\n")
    custom_metrics = None


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY — {split.upper()}  (max 1000)")
print(f"{'='*60}")
print(f"  {'Model':<28} {'Top-1':>6}  {'Top-5':>6}  {'Top-10':>7}  {'Score':>7}")
print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")

rows = [
    ('InsightFace buffalo_l',    insightface_metrics),
    ('CLIP ViT-B/32 (zero-shot)', clip_metrics),
    ('Custom ResNet50+ArcFace',  custom_metrics),
]
for name, m in rows:
    if m is None:
        print(f"  {name:<28}  {'(skipped/failed)':>28}")
    else:
        print(f"  {name:<28}  {m['top1']*100:5.1f}%  {m['top5']*100:5.1f}%  {m['top10']*100:6.1f}%  {m['score']:7.1f}")

print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
if args.visualize:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import random

    # collect only models that actually ran
    model_results = {
        'InsightFace': insightface_results,
        'CLIP':        clip_results,
        'Custom':      custom_results,
    }
    active = {k: v for k, v in model_results.items() if v is not None}

    if not active:
        print("No model results to visualise.\n")
    else:
        N_VIS  = args.n_vis
        TOP_K  = args.top_vis

        # pick N_VIS random queries (same set for all models)
        random.seed(42)
        sample_queries = random.sample(q_fnames, min(N_VIS, len(q_fnames)))

        n_models = len(active)
        # layout: one block per model, each block = N_VIS rows × (1 + TOP_K) cols
        # stack model blocks vertically, separated by a blank row
        COLS = 1 + TOP_K
        ROWS = n_models * N_VIS + (n_models - 1)   # blank separator rows between models

        fig, axes = plt.subplots(
            ROWS, COLS,
            figsize=(COLS * 2.0, ROWS * 2.2),
        )
        # always 2-D
        if ROWS == 1:
            axes = axes[np.newaxis, :]

        # turn off all axes up front
        for ax in axes.flatten():
            ax.axis('off')

        def load_img(path):
            try:
                return np.array(Image.open(path).convert('RGB'))
            except Exception:
                return np.zeros((112, 112, 3), dtype=np.uint8)

        # border colours: green = correct match, red = wrong
        def border_color(q_fname, g_fname):
            return '#2ecc71' if get_identity(q_fname) == get_identity(g_fname) else '#e74c3c'

        row = 0
        for m_idx, (model_name, results) in enumerate(active.items()):
            if m_idx > 0:
                row += 1   # blank separator row

            for qi, q_fname in enumerate(sample_queries):
                ax_row = row + qi

                # ── query image (leftmost column) ──────────────────────────
                q_path = os.path.join(QUERY_DIR, q_fname)
                ax = axes[ax_row][0]
                ax.imshow(load_img(q_path))
                ax.axis('off')
                ax.set_title(
                    f'{model_name}\n{q_fname}' if qi == 0 else q_fname,
                    fontsize=6, pad=2, color='#222222'
                )
                # gold border for query
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('#f39c12')
                    spine.set_linewidth(2.5)

                # ── retrieved images ───────────────────────────────────────
                retrieved = results.get(q_fname, [])[:TOP_K]
                for k, g_fname in enumerate(retrieved):
                    g_path = os.path.join(GALLERY_DIR, g_fname)
                    ax = axes[ax_row][k + 1]
                    ax.imshow(load_img(g_path))
                    ax.axis('off')
                    color = border_color(q_fname, g_fname)
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(color)
                        spine.set_linewidth(2.5)
                    ax.set_xlabel(
                        g_fname, fontsize=5, labelpad=2,
                        color='#333333'
                    )

            row += N_VIS

        # column headers on the very first row
        axes[0][0].set_title('QUERY', fontsize=7, fontweight='bold', pad=4)
        for k in range(TOP_K):
            axes[0][k+1].set_title(f'Top-{k+1}', fontsize=7, fontweight='bold', pad=4)

        # legend
        correct_patch = mpatches.Patch(color='#2ecc71', label='Correct identity')
        wrong_patch   = mpatches.Patch(color='#e74c3c', label='Wrong identity')
        query_patch   = mpatches.Patch(color='#f39c12', label='Query')
        fig.legend(
            handles=[query_patch, correct_patch, wrong_patch],
            loc='lower center', ncol=3, fontsize=7,
            framealpha=0.8, bbox_to_anchor=(0.5, 0.0)
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out_png = os.path.join(RESULTS_DIR, f'retrieval_comparison_v3_{split}.png')
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualisation saved → {out_png}\n")
