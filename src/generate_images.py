from pathlib import Path
from PIL import Image
from src.data import CUBDataset

output_dir = Path("image_examples")
output_dir.mkdir(exist_ok=True)

conditions = ["original", "bbox", "seg_fg", "bg_swap", "bg_only"]

for condition in conditions:
    dataset = CUBDataset(split="test", condition=condition)

    img, label, img_id = dataset[0] #change index for different images

    # If transform converts it to a tensor, convert it back to PIL.
    if not isinstance(img, Image.Image):
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(img)

    img.save(output_dir / f"{condition}.png")

    print(f"Saved {condition}.png")

print(f"\nImages saved in: {output_dir.resolve()}")