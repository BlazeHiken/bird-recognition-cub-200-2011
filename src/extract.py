import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import config
from src.data import CUBDataset
from src.models import load_backbone

def extract_features(model_name="resnet50", condition="original"):
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {model_name}...")
    model, preprocess, feat_dim = load_backbone(model_name)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    out_dir = config.FEATURES_DIR / model_name
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for split in ["train", "test"]:
        print(f"Extracting features for {split} split, condition: {condition}...")
        dataset = CUBDataset(split=split, condition=condition, transform=preprocess)
        loader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=config.NUM_WORKERS
        )
        
        all_features = []
        all_labels = []
        all_ids = []
        
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
            for imgs, labels, img_ids in tqdm(loader):
                imgs = imgs.to(device)
                features = model(imgs)
                all_features.append(features.cpu())
                all_labels.append(labels)
                all_ids.append(img_ids)
                
        extract_time = time.time() - start_time
        num_images = len(dataset)
        latency_per_img = extract_time / num_images
        
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        print(f"Extracted {num_images} images in {extract_time:.2f}s")
        print(f"Latency per image: {latency_per_img*1000:.2f} ms")
        if peak_mem > 0:
            print(f"Peak GPU memory: {peak_mem:.2f} MB")
            
        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        ids_tensor = torch.cat(all_ids, dim=0)
        
        save_path = out_dir / f"{condition}_{split}.pt"
        torch.save({
            "features": features_tensor,
            "labels": labels_tensor,
            "img_ids": ids_tensor,
            "param_count": param_count,
            "extract_time": extract_time,
            "latency_per_img_ms": latency_per_img * 1000,
            "peak_mem_mb": peak_mem
        }, save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--condition", type=str, default="original")
    args = parser.parse_args()
    extract_features(args.model, args.condition)
