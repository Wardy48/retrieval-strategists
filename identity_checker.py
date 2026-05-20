import os
import glob
import re
from collections import defaultdict

def analyze_dataset_domains(train_dir):
    print(f"[*] Analizzando la directory: {train_dir}...")
    
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(train_dir, ext)))
        
    if not image_paths:
        print("[!] Nessuna immagine trovata. Controlla il percorso.")
        return

    # Struttura: { "10177": {"celebrities": 2, "cartoons": 0, "synthetic": 1}, ... }
    identity_stats = defaultdict(lambda: {"celebrities": 0, "cartoons": 0, "synthetic": 0, "other": 0})
    
    for path in image_paths:
        filename = os.path.basename(path).lower()
        
        # Estrai ID
        match = re.search(r'\d+', filename)
        if not match:
            continue
        identity = match.group(0)
        
        # Classifica il dominio in base al nome file
        if "celebrities" in filename:
            identity_stats[identity]["celebrities"] += 1
        elif "cartoons" in filename:
            identity_stats[identity]["cartoons"] += 1
        elif "synthetic" in filename:
            identity_stats[identity]["synthetic"] += 1
        else:
            identity_stats[identity]["other"] += 1

    # --- Calcolo delle Statistiche ---
    total_ids = len(identity_stats)
    
    # Contatori per le combinazioni
    only_real = 0
    only_fake = 0  # Solo cartoons e/o synthetic
    real_and_synthetic = 0
    real_and_cartoons = 0
    all_domains = 0
    
    for uid, stats in identity_stats.items():
        has_real = stats["celebrities"] > 0
        has_synth = stats["synthetic"] > 0
        has_cart = stats["cartoons"] > 0
        
        if has_real and not (has_synth or has_cart):
            only_real += 1
        elif not has_real and (has_synth or has_cart):
            only_fake += 1
            
        if has_real and has_synth:
            real_and_synthetic += 1
        if has_real and has_cart:
            real_and_cartoons += 1
        if has_real and has_synth and has_cart:
            all_domains += 1

    # --- Output ---
    print("\n==========================================")
    print("      DATASET DOMAIN ASSUMPTIONS REPORT")
    print("==========================================")
    print(f"Totale Immagini: {len(image_paths)}")
    print(f"Totale ID Unici: {total_ids}")
    print("------------------------------------------")
    print(f"ID ESCLUSIVAMENTE 'Reali' (solo celebrities): {only_real}")
    print(f"ID ESCLUSIVAMENTE 'Finti' (solo cartoons/synthetic): {only_fake}")
    print("------------------------------------------")
    print(f"CROSS-DOMAIN OVERLAP (La nostra speranza):")
    print(f"ID con Reale + Sintetico: {real_and_synthetic}")
    print(f"ID con Reale + Cartoons: {real_and_cartoons}")
    print(f"ID presenti in TUTTI i domini: {all_domains}")
    print("==========================================\n")
    
    if real_and_synthetic == 0 and real_and_cartoons == 0:
        print("[!] ALLARME: Il tuo dataset di training NON ha sovrapposizioni cross-domain.")
        print("[!] La rete non ha modo di imparare che una foto reale e una finta sono la stessa persona.")

if __name__ == "__main__":
    TRAIN_DIR = "/home/disi/retrivial_strategist/dataset_final_flat/train/all_images"
    analyze_dataset_domains(TRAIN_DIR)