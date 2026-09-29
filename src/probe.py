import csv
import torch
import torch.nn as nn
from src import config

def train_probe(model_name="resnet50", condition="original", num_epochs=100):
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(config.SEED)
    
    train_path = config.FEATURES_DIR / model_name / f"{condition}_train.pt"
    test_path = config.FEATURES_DIR / model_name / f"{condition}_test.pt"
    
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    X_train_full = train_data["features"].float()
    y_train_full = train_data["labels"]
    X_test = test_data["features"].float()
    y_test = test_data["labels"]
    
    # Standardize features
    mean = X_train_full.mean(dim=0, keepdim=True)
    std = X_train_full.std(dim=0, keepdim=True) + 1e-8
    
    X_train_full = (X_train_full - mean) / std
    X_test = (X_test - mean) / std
    
    X_train_full = X_train_full.to(device)
    y_train_full = y_train_full.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Create validation split from train data
    dataset_size = len(X_train_full)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    
    feat_dim = X_train.shape[1]
    num_classes = 200
    
    lrs = [1e-3, 5e-4, 1e-4]
    best_val_acc = 0.0
    best_lr = lrs[0]
    
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Learning Rate sweep...")
    for lr in lrs:
        model = nn.Linear(feat_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(X_train.size()[0])
            for i in range(0, X_train.size()[0], config.BATCH_SIZE):
                batch_indices = permutation[i:i+config.BATCH_SIZE]
                batch_x, batch_y = X_train[batch_indices], y_train[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()
            
        print(f"LR: {lr}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
            
    print(f"Best LR: {best_lr}. Retraining on full train set...")
    
    final_model = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_lr)
    
    for epoch in range(num_epochs):
        final_model.train()
        permutation = torch.randperm(X_train_full.size()[0])
        for i in range(0, X_train_full.size()[0], config.BATCH_SIZE):
            batch_indices = permutation[i:i+config.BATCH_SIZE]
            batch_x, batch_y = X_train_full[batch_indices], y_train_full[batch_indices]
            
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    final_model.eval()
    with torch.no_grad():
        test_outputs = final_model(X_test)
        test_preds = test_outputs.argmax(dim=1)
        test_acc = (test_preds == y_test).float().mean().item()
        
    print(f"Final Test Top-1 Acc: {test_acc:.4f}")
    
    csv_path = config.RESULTS_DIR / "results.csv"
    file_exists = csv_path.exists()
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Condition", "Top1", "Params", "ExtractTime(s)", "LatencyPerImg(ms)", "PeakMem(MB)", "BestLR"])
        writer.writerow([
            model_name,
            condition,
            f"{test_acc:.4f}",
            train_data["param_count"],
            f"{train_data['extract_time']:.2f}",
            f"{train_data['latency_per_img_ms']:.2f}",
            f"{train_data['peak_mem_mb']:.2f}",
            best_lr
        ])
    print(f"Results appended to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--condition", type=str, default="original")
    args = parser.parse_args()
    train_probe(args.model, args.condition)
