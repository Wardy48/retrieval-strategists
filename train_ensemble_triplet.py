import os
# Fix memory fragmentation before PyTorch initializes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================================
# 1. TRIPLET DATASET LOADER (ROBUST VERSION)
# ==========================================
class TripletIdentityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Safe multi-extension search to match any dataset format
        self.image_paths = []
        valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        for ext in valid_extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        
        # Group all images by their unique numerical identity string using Regex
        self.identity_to_paths = {}
        for path in self.image_paths:
            match = re.search(r'\d+', os.path.basename(path))
            if match:
                identity = match.group(0)
                if identity not in self.identity_to_paths:
                    self.identity_to_paths[identity] = []
                self.identity_to_paths[identity].append(path)
                
        self.identities = list(self.identity_to_paths.keys())
        print(f"[+] Formatted Triplet Dataset: Found {len(self.identities)} unique identities across {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        
        # Robust ID extraction
        match = re.search(r'\d+', os.path.basename(anchor_path))
        anchor_id = match.group(0) if match else self.identities[0]
        
        # 1. Identifica il dominio dell'Anchor ("celebrities" = Reale, altrimenti Sintetico/Cartone)
        is_anchor_real = "celebrities" in anchor_path.lower()
        
        pos_candidates = self.identity_to_paths[anchor_id]
        
        # 2. CROSS-DOMAIN POSITIVE MINING: Forza la rete a cambiare dominio!
        cross_domain_positives = []
        for p in pos_candidates:
            if p == anchor_path: continue
            is_p_real = "celebrities" in p.lower()
            # Se l'anchor è reale, vogliamo un positivo finto (e viceversa)
            if is_anchor_real != is_p_real:
                cross_domain_positives.append(p)
                
        # Usa il positivo di dominio opposto se esiste, altrimenti accontentati (fallback)
        if len(cross_domain_positives) > 0:
            pos_path = random.choice(cross_domain_positives)
        elif len(pos_candidates) > 1:
            pos_path = random.choice([p for p in pos_candidates if p != anchor_path])
        else:
            pos_path = anchor_path
            
        # 3. CROSS-DOMAIN NEGATIVE MINING: Il negativo deve appartenere allo stesso dominio del positivo
        is_pos_real = "celebrities" in pos_path.lower()
        neg_id = random.choice([i for i in self.identities if i != anchor_id])
        neg_candidates = self.identity_to_paths[neg_id]
        
        target_domain_negatives = []
        for n in neg_candidates:
            is_n_real = "celebrities" in n.lower()
            # Il negativo deve imitare lo stile del positivo, per costringere la rete 
            # a non usare lo stile per risolvere il task, ma a cercare l'identità.
            if is_pos_real == is_n_real:
                target_domain_negatives.append(n)
                
        if len(target_domain_negatives) > 0:
            neg_path = random.choice(target_domain_negatives)
        else:
            neg_path = random.choice(neg_candidates)
            
        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        neg_img = Image.open(neg_path).convert("RGB")
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            
        return anchor_img, pos_img, neg_img

# ==========================================
# 2. MODIFIED CNN WITH L2 EMBEDDING HEAD
# ==========================================
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
        
        cat = torch.cat((c1, c2, c3), dim=1)
        emb = self.fc(cat)
        
        # CRITICAL FIX: L2 Normalize to project embeddings onto an hypersphere
        return F.normalize(emb, p=2, dim=1)

def build_ensemble_models(embedding_dim=256):
    # 1. MobileNetV2
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mod_mobilenet = ModifiedCNN(mobilenet.features, in_channels=1280, embedding_dim=embedding_dim)
    
    # 2. DenseNet201 (Optimized with memory_efficient=True to prevent OOM)
    densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT, memory_efficient=True)
    mod_densenet = ModifiedCNN(densenet.features, in_channels=1920, embedding_dim=embedding_dim)
    
    # 3. Vision Transformer (ViT)
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    vit.heads.head = nn.Linear(vit.heads.head.in_features, embedding_dim)
    
    return mod_mobilenet, mod_densenet, vit

# ==========================================
# 3. ROBUST TRAINING FUNCTION
# ==========================================
def train_metric_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = "/home/disi/retrivial_strategist/dataset_final_flat/train/all_images"
    dataset = TripletIdentityDataset(root_dir=train_dir, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    EMBEDDING_DIM = 256
    model_mn, model_dn, model_vit = build_ensemble_models(embedding_dim=EMBEDDING_DIM)
    model_mn.to(device)
    model_dn.to(device)
    model_vit.to(device)
    
    # Standard triplet loss works flawlessly now that vectors are mapped on a sphere
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    optimizer_mn = optim.Adam(model_mn.parameters(), lr=5e-5)
    optimizer_dn = optim.Adam(model_dn.parameters(), lr=5e-5)
    optimizer_vit = optim.Adam(model_vit.parameters(), lr=2e-5)
    
    scaler = torch.cuda.amp.GradScaler()
    
    epochs = 10
    print("[*] Commencing Optimized Ensemble Metric Learning Training...")
    
    for epoch in range(epochs):
        model_mn.train()
        model_dn.train()
        model_vit.train()
        total_loss = 0.0
        
        for batch_idx, (anc_img, pos_img, neg_img) in enumerate(dataloader):
            anc_img, pos_img, neg_img = anc_img.to(device), pos_img.to(device), neg_img.to(device)
            
            # --- MobileNetV2 Pipeline (CNN handles internal L2 Normalization) ---
            optimizer_mn.zero_grad()
            with torch.cuda.amp.autocast():
                out_mn_a = model_mn(anc_img)
                out_mn_p = model_mn(pos_img)
                out_mn_n = model_mn(neg_img)
                loss_mn = criterion(out_mn_a, out_mn_p, out_mn_n)
            scaler.scale(loss_mn).backward()
            scaler.step(optimizer_mn)
            
            # --- DenseNet201 Pipeline (CNN handles internal L2 Normalization) ---
            optimizer_dn.zero_grad()
            with torch.cuda.amp.autocast():
                out_dn_a = model_dn(anc_img)
                out_dn_p = model_dn(pos_img)
                out_dn_n = model_dn(neg_img)
                loss_dn = criterion(out_dn_a, out_dn_p, out_dn_n)
            scaler.scale(loss_dn).backward()
            scaler.step(optimizer_dn)
            
            # --- ViT Pipeline (L2 Normalized directly in training loop) ---
            optimizer_vit.zero_grad()
            with torch.cuda.amp.autocast():
                # Enforce L2 Normalization explicitly for ViT embeddings
                out_vit_a = F.normalize(model_vit(anc_img), p=2, dim=1)
                out_vit_p = F.normalize(model_vit(pos_img), p=2, dim=1)
                out_vit_n = F.normalize(model_vit(neg_img), p=2, dim=1)
                loss_vit = criterion(out_vit_a, out_vit_p, out_vit_n)
            scaler.scale(loss_vit).backward()
            scaler.step(optimizer_vit)
            
            scaler.update()
            
            combined_batch_loss = loss_mn.item() + loss_dn.item() + loss_vit.item()
            total_loss += combined_batch_loss
            
            if batch_idx % 50 == 0:
                print(f"   Batch [{batch_idx}/{len(dataloader)}] | Combined Triplet Loss: {combined_batch_loss:.4f}")
                
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Epoch Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("ensemble_checkpoints", exist_ok=True)
    torch.save(model_mn.state_dict(), "ensemble_checkpoints/mod_mobilenet.pth")
    torch.save(model_dn.state_dict(), "ensemble_checkpoints/mod_densenet.pth")
    torch.save(model_vit.state_dict(), "ensemble_checkpoints/vit.pth")
    print("\n[+] Metric models saved successfully without OOM exceptions.")

if __name__ == "__main__":
    train_metric_ensemble()