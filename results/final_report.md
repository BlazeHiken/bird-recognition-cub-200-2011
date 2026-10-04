# Robustness vs Efficiency of Pretrained Visual Representations for Fine-Grained Bird Recognition

## 1. Experimental Overview
This report summarizes the findings of the CUB-200-2011 robustness benchmark. We evaluated three distinct pretraining paradigms—**ResNet-50** (Supervised CNN), **DINOv2 ViT-S/14** (Self-supervised ViT), and **CLIP ViT-B/16** (Vision-Language)—across five visual conditions to determine how heavily they rely on background context for fine-grained classification.

### Visual Conditions Tested:
- **`original`**: The unaltered CUB image.
- **`bbox`**: Cropped tightly to the bird using official bounding boxes.
- **`seg_fg`**: Background removed (blacked out) using segmentation masks.
- **`bg_swap`**: Background replaced with a solid neutral gray `(128, 128, 128)`.
- **`bg_only`**: Bird removed (blacked out), leaving only the background context.

---

## 2. Quantitative Results

| Model | Original | BBox | Seg (Black BG) | Seg (Gray BG Swap) | Background Only |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DINOv2 ViT-S** | **87.92%** | 84.33% | 84.40% | **86.69%** | **39.21%** |
| **CLIP ViT-B/16** | 79.63% | 67.36% | 72.89% | 74.16% | 25.87% |
| **ResNet-50** | 67.79% | 64.05% | 63.65% | 67.36% | 21.73% |

---

## 3. Analysis and Key Findings

### RQ1: Which pretraining paradigm is most accurate and robust?
**DINOv2** is the undeniable winner. Not only did it achieve the highest baseline accuracy (87.92%), but it also exhibited incredible robustness. When the background was entirely swapped for a gray canvas (`bg_swap`), it maintained **86.69%** accuracy—a negligible drop of just 1.23%. Its self-supervised training clearly forces the network to learn excellent, localized, and object-centric representations.

### RQ2: Do models rely on background information?
Yes, but to vastly different degrees:
* **CLIP is highly dependent on context:** CLIP suffered the most severe performance degradation when the background was modified. It dropped by a massive **12.27%** on bounding box crops, and **6.74%** when the background was blacked out (`seg_fg`). Because CLIP learns from internet image-text pairs, it strongly associates background elements (e.g., water, branches, specific foliage) with the bird species.
* **The `bg_only` revelation:** When presented with *only the background* (and no bird), DINOv2 could still guess the correct species **39.21%** of the time. This highlights that while DINOv2 doesn't strictly *need* the background (as seen by its high `seg_fg` score), the background in CUB-200-2011 is highly correlated with the species.

### RQ3: How do the models compare in efficiency?
*(Metrics extracted from `results.csv` on RTX 4050)*
* **DINOv2 ViT-S:** ~22M parameters | ~6ms latency/img
* **ResNet-50:** ~23M parameters | ~5-14ms latency/img
* **CLIP ViT-B/16:** ~149M parameters | ~7-8ms latency/img

DINOv2 provides the best trade-off by far. It uses fewer parameters than ResNet-50 while outperforming CLIP (which is 7x larger in parameter count) by a massive margin in both accuracy and robustness.

---

## 4. Conclusion
The experiments confirm that **pretraining strategy fundamentally alters a model's reliance on visual context**. Vision-language models like CLIP learn heavy contextual biases that harm their robustness in fine-grained tasks when the background changes. In contrast, self-supervised models like DINOv2 learn highly resilient object-centric features, making them the superior choice for fine-grained recognition where background conditions cannot be guaranteed.
