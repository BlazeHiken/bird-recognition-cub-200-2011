# Bird Species Recognition — CUB-200-2011

Robustness vs Efficiency of Pretrained Visual Representations for Fine-Grained Bird Recognition.

This project evaluates how different pretrained visual representation models perform on fine-grained bird species classification under controlled changes to the visual context, using the [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset.

---

## Research Question

> Do different pretrained visual representations rely differently on background information when performing fine-grained bird species recognition?

## Models

| Model | Pretraining | Parameters |
| :--- | :--- | ---: |
| ResNet-50 | Supervised (ImageNet) | 23.5M |
| DINOv2 ViT-S/14 | Self-supervised | 22.1M |
| CLIP ViT-B/16 | Vision-language (OpenAI) | 149.6M |

## Visual Conditions

Each model is evaluated on the same test images under five conditions:

| Condition | Description |
| :--- | :--- |
| `original` | Unaltered CUB image |
| `bbox` | Cropped to the bird using official bounding boxes |
| `seg_fg` | Background removed (black) using segmentation masks |
| `bg_swap` | Background replaced with solid gray (128, 128, 128) |
| `bg_only` | Bird removed, only background remains |

## Results

### Top-1 Accuracy (%)

| Model | Original | BBox | Seg FG | BG Swap | BG Only |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DINOv2 ViT-S/14 | **87.92** | **84.33** | **84.40** | **86.69** | **39.21** |
| CLIP ViT-B/16 | 79.63 | 67.36 | 72.89 | 74.16 | 25.87 |
| ResNet-50 | 67.79 | 64.05 | 63.65 | 67.36 | 21.73 |

### Accuracy Drop (Original vs Seg FG)

| Model | Drop |
| :--- | :---: |
| DINOv2 ViT-S/14 | -3.52% |
| ResNet-50 | -4.14% |
| CLIP ViT-B/16 | -6.74% |

### Efficiency (RTX 4050 Laptop GPU, 6 GB VRAM)

| Model | Latency/Image | Peak GPU Memory |
| :--- | :---: | :---: |
| DINOv2 ViT-S/14 | ~6 ms | 369 MB |
| ResNet-50 | ~5-14 ms | 494 MB |
| CLIP ViT-B/16 | ~7-8 ms | 1014 MB |

## Key Findings

1. **DINOv2 is the most accurate and robust.** It achieves the highest accuracy across all conditions and barely flinches when the background is swapped (-1.23%).
2. **CLIP relies heavily on background context.** It suffers the largest accuracy drops when background information is removed or altered, suggesting vision-language pretraining encodes strong contextual biases.
3. **Background is predictive of species.** Even with the bird completely removed (`bg_only`), DINOv2 can still classify the correct species 39% of the time, highlighting strong habitat-species correlations in CUB-200-2011.

## Methodology

All models use **frozen pretrained backbones** as feature extractors. A single **linear classification head** is trained on the cached features using AdamW with a learning rate sweep over three values (`1e-3`, `5e-4`, `1e-4`) on a 80/20 train/validation split. The best learning rate is then used to retrain on the full training set and evaluated on the official CUB test split.

Features are extracted in eval mode with FP16 autocast.

## Project Structure

```
bird-recognition-cub-200-2011/
├── src/
│   ├── config.py        # Paths, seeds, batch size
│   ├── data.py          # CUB dataset loader (5 visual conditions)
│   ├── models.py        # Backbone loading (ResNet-50, DINOv2, CLIP)
│   ├── extract.py       # Frozen feature extraction to .pt files
│   └── probe.py         # Linear probe training and evaluation
├── PRD/
│   ├── prdv1.md         # Baseline PRD
│   └── prdv2.md         # Robustness benchmark PRD
├── results/
│   ├── results.csv      # All experimental results
│   └── final_report.md  # Detailed analysis
├── features/            # Cached feature tensors (gitignored)
├── CUB_200_2011/        # Dataset (gitignored)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support

### Installation

```bash
git clone https://github.com/BlazeHiken/bird-recognition-cub-200-2011.git
cd bird-recognition-cub-200-2011
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

Download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract it so that `CUB_200_2011/` sits in the project root. The segmentation masks should be placed inside `CUB_200_2011/segmentations/`.

## Usage

### 1. Extract features

```bash
python -m src.extract --model resnet50 --condition original
python -m src.extract --model dinov2_vits14 --condition original
python -m src.extract --model clip_vit_b16 --condition original
```

Available models: `resnet50`, `dinov2_vits14`, `clip_vit_b16`

Available conditions: `original`, `bbox`, `seg_fg`, `bg_swap`, `bg_only`

### 2. Train linear probe and evaluate

```bash
python -m src.probe --model resnet50 --condition original
python -m src.probe --model dinov2_vits14 --condition original
python -m src.probe --model clip_vit_b16 --condition original
```

Results are appended to `results/results.csv`.

## Limitations

- **ImageNet overlap:** CUB-200-2011 images overlap with ImageNet. Models pretrained on ImageNet (ResNet-50) may have seen some test images during pretraining.
- **Linear probe only:** Results reflect the quality of frozen representations. Full fine-tuning may change the relative rankings.
- **Single seed:** Experiments use a fixed seed (42) without repeated runs or confidence intervals.

## License

This project uses the [CUB-200-2011 dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) for academic research purposes.
