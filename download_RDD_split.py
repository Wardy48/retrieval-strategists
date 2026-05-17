import os
import zipfile
import shutil
import hashlib
import random
import urllib.request
from pyspark import SparkContext, SparkConf

# Set random seed for reproducibility
random.seed(42)

def extract_zip(zip_path, extract_to):
    print(f"[*] Extracting '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[+] Extraction complete!")
    
    # Diagnostic check: See what files are physically present on disk
    print("[*] Inspecting extracted directory structure on disk...")
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if not file.startswith('.'):
                extracted_files.append(os.path.join(root, file))
    print(f"[->] Python disk-scan detected {len(extracted_files)} files extracted on the VM.")

def process_file_item(item, broadcast_map_val, absolute_output_dir):
    """
    Worker-side function executed in parallel across Spark partitions.
    Processes a single image file (path and binary content).
    """
    path, content = item
    filename = os.path.basename(path)
    
    # Ignore OS metadata/hidden files
    if not filename or filename.startswith('.'):
        return
        
    # 1. Process Cartoons
    if filename.startswith('c'):
        hash_score = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 100
        split = 'train' if hash_score < 80 else ('val' if hash_score < 90 else 'test')
        dest_dir = os.path.join(absolute_output_dir, split, 'cartoons')
        
    # 2. Process Synthetic
    elif filename.startswith('s'):
        hash_score = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 100
        split = 'train' if hash_score < 80 else ('val' if hash_score < 90 else 'test')
        dest_dir = os.path.join(absolute_output_dir, split, 'synthetic')
        
    # 3. Process Celebrities (Identity Preserving Map)
    elif '_' in filename:
        identity_id = filename.split('_')[0]
        if identity_id.isdigit():
            split = broadcast_map_val.get(identity_id, 'test')
            dest_dir = os.path.join(absolute_output_dir, split, 'celebrities', identity_id)
        else:
            return
    else:
        return # Skip unknown file schemas

    # Ensure targeted folder structure exists and write raw bytes
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, filename), 'wb') as f:
        f.write(bytes(content))

def process_and_split_dataset_rdd(source_dir, output_dir):
    print("[*] Initializing PySpark Context for distributed transformation...")
    conf = SparkConf().setAppName("CelebrityDatasetSplitter").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf=conf)
    
    # FIX 1: Explicitly tell Spark's Hadoop file reader to look inside subdirectories recursively
    sc._jsc.hadoopConfiguration().set("mapreduce.input.fileinputformat.input.dir.recursive", "true")
    
    absolute_output_dir = os.path.abspath(output_dir)
    
    print("[*] Loading raw image binary streams into an RDD...")
    # Prepend 'file://' prefix to prevent absolute path namespace confusion in local Hadoop context
    binary_rdd = sc.binaryFiles("file://" + os.path.abspath(source_dir))
    
    total_files = binary_rdd.count()
    print(f"[->] Total raw files detected inside Spark RDD: {total_files}")
    if total_files == 0:
        print("[!] ERROR: Spark found 0 files. The archive might be structured unexpectedly.")
        sc.stop()
        return

    print("[*] Scanning dataset RDD to isolate unique celebrity identity numbers...")
    def get_celebrity_id(item):
        filename = os.path.basename(item[0])
        if '_' in filename and not filename.startswith(('c', 's')):
            parts = filename.split('_')
            if parts[0].isdigit():
                return parts[0]
        return None

    celebrity_ids = (binary_rdd
                     .map(get_celebrity_id)
                     .filter(lambda x: x is not None)
                     .distinct()
                     .collect())
                     
    print(f"[->] Detected {len(celebrity_ids)} distinct celebrity identities via RDD analysis.")
    
    if len(celebrity_ids) == 0:
        print("[!] ERROR: Found files but 0 matched the celebrity identity pattern (e.g. '5472_4.jpg')")
        sc.stop()
        return

    # Perform open-set partitioning directly on the driver context
    random.shuffle(celebrity_ids)
    n_total = len(celebrity_ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_ids = set(celebrity_ids[:n_train])
    val_ids = set(celebrity_ids[n_train:n_train+n_val])
    
    identity_split_map = {}
    for cid in celebrity_ids:
        if cid in train_ids:
            identity_split_map[cid] = 'train'
        elif cid in val_ids:
            identity_split_map[cid] = 'val'
        else:
            identity_split_map[cid] = 'test'
            
    # Optimize data transfer: Broadcast the identity assignment table to all Spark workers
    broadcast_map = sc.broadcast(identity_split_map)
    map_val = broadcast_map.value
    
    print("[*] Executing parallel sorting and local deployment using Spark tasks...")
    binary_rdd.foreach(lambda item: process_file_item(item, map_val, absolute_output_dir))
    
    sc.stop()
    print(f"[+] Multi-threaded processing successfully finalized inside '{absolute_output_dir}/'")

if __name__ == "__main__":
    # Update this with your actual GitHub Release download URL link
    DOWNLOAD_URL = "https://github.com/Wardy48/retrieval-strategists/releases/latest/download/retrieval_strategists_full_dataset_in_partitions_v2.zip"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ZIP_FILE_PATH = os.path.join(script_dir, "dataset_downloaded.zip")
    EXTRACTED_DIR = os.path.join(script_dir, "dataset_extracted")
    FINAL_SPLIT_DIR = os.path.join(script_dir, "dataset_final")
    
    try:
        if not os.path.exists(ZIP_FILE_PATH):
            print(f"[*] Downloading dataset directly from GitHub Release...")
            urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_FILE_PATH)
            print(f"[+] Download complete!")
        
        extract_zip(ZIP_FILE_PATH, EXTRACTED_DIR)
        
        # Parallel computation phase
        process_and_split_dataset_rdd(EXTRACTED_DIR, FINAL_SPLIT_DIR)
        
        print("[*] Purging raw unzipped files from storage...")
        if os.path.exists(ZIP_FILE_PATH):
            os.remove(ZIP_FILE_PATH)
        if os.path.exists(EXTRACTED_DIR):
            shutil.rmtree(EXTRACTED_DIR)
        print("[+] Process successfully completed!")
        
    except Exception as e:
        print(f"[-] Execution failure: {e}")