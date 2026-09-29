import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src import config

class CUBDataset(Dataset):
    def __init__(self, split="train", condition="original", transform=None):
        self.split = split
        self.condition = condition
        self.transform = transform
        
        # Parse metadata
        images = pd.read_csv(config.DATA_DIR / "images.txt", sep=" ", names=["img_id", "path"])
        labels = pd.read_csv(config.DATA_DIR / "image_class_labels.txt", sep=" ", names=["img_id", "label"])
        split_df = pd.read_csv(config.DATA_DIR / "train_test_split.txt", sep=" ", names=["img_id", "is_train"])
        bboxes = pd.read_csv(config.DATA_DIR / "bounding_boxes.txt", sep=" ", names=["img_id", "x", "y", "w", "h"])
        
        df = images.merge(labels, on="img_id").merge(split_df, on="img_id").merge(bboxes, on="img_id")
        
        is_train_flag = 1 if split == "train" else 0
        self.data = df[df["is_train"] == is_train_flag].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row["img_id"]
        img_path = config.DATA_DIR / "images" / row["path"]
        label = int(row["label"]) - 1 # 0-indexed labels
        
        img = Image.open(img_path).convert("RGB")
        
        if self.condition == "original":
            pass # Keep original image
        elif self.condition == "bbox":
            pass # Implement later
        elif self.condition == "seg_fg":
            pass # Implement later
        elif self.condition == "bg_only":
            pass # Implement later
        elif self.condition == "bg_swap":
            pass # Implement later
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_id
