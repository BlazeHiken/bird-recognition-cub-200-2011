# PRD v0.2 — Robustness and Efficiency Benchmark for Fine-Grained Bird Recognition

## 1. Project Overview

### Project Title

**Robustness vs Efficiency of Pretrained Visual Representations for Fine-Grained Bird Recognition**

### Dataset

Caltech-UCSD Birds-200-2011 (CUB-200-2011)

### Objective

Evaluate how different pretrained visual representation models perform for fine-grained bird species classification under controlled changes to the visual context.

The study will compare:

- Classification accuracy
- Robustness to background/context changes
- Computational cost
- Performance in limited-data settings

The goal is to determine whether differences in pretraining strategy and model architecture affect reliance on background information when recognizing visually similar bird species.

---

## 2. Research Problem

Fine-grained bird recognition requires distinguishing species with subtle visual differences.

A model may achieve high classification accuracy while relying partly on contextual or background information rather than bird-specific visual features.

This creates an important question:

> **Do different pretrained visual representations rely differently on background information when performing fine-grained bird species recognition?**

The project will investigate this through controlled evaluation using the segmentation annotations provided for CUB-200-2011.

---

## 3. Research Questions

### RQ1 — Accuracy

How accurately can different pretrained visual representation models classify the 200 CUB bird species?

### RQ2 — Robustness

How does classification performance change when background information is removed or altered?

### RQ3 — Pretraining

Do different pretraining approaches exhibit different levels of robustness to background changes?

### RQ4 — Efficiency

How do accuracy and robustness compare against computational cost?

### RQ5 — Limited Data

How do the models perform when only a small number of training examples per species are available?

---

## 4. Dataset

### CUB-200-2011

The dataset contains:

- 11,788 images
- 200 bird species
- Official train/test split
- Bounding-box annotations
- 15 body-part annotations
- 312 visual attributes

Additionally, the separate CUB segmentation release has been added to the project.

### Current Dataset Structure

```text
CUB_200_2011/
├── attributes/
├── images/
├── parts/
├── segmentations/
├── bounding_boxes.txt
├── classes.txt
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── README
```

The official CUB train/test split will be retained for the main experiments.

---

## 5. Experimental Data Conditions

The same test images will be evaluated under multiple visual conditions.

### Condition A — Original

The original CUB image.

```text
Original Image
      ↓
Model
      ↓
Prediction
```

This establishes the standard classification result.

### Condition B — Bounding-Box Crop

The provided CUB bounding box will be used to crop the image around the bird.

```text
Original Image
      ↓
Bounding Box Crop
      ↓
Model
      ↓
Prediction
```

This reduces irrelevant image background while retaining the bird and some surrounding context.

### Condition C — Segmentation / Background Removal

The segmentation mask will be used to isolate the bird from the surrounding background.

```text
Original Image
      +
Segmentation Mask
      ↓
Bird-Isolated Image
      ↓
Model
      ↓
Prediction
```

This provides a more controlled test of how much background information contributes to recognition.

### Condition D — Background Replacement

Where computationally and methodologically practical, the segmentation mask will be used to replace the original background with a controlled alternative background.

The bird itself will remain unchanged.

```text
Original Image
        +
Segmentation Mask
        +
Replacement Background
        ↓
Background-Modified Image
        ↓
Model
        ↓
Prediction
```

The exact background-replacement protocol will be finalized before experimentation to ensure that the manipulation does not introduce unintended artifacts.

---

## 6. Models

The study will compare pretrained visual representation models from different model/pretraining families.

### Initial Candidate Models

**Model 1 — ResNet-50**  
A supervised ImageNet-pretrained CNN.

**Model 2 — EfficientNet**  
A computationally efficient supervised CNN.

**Model 3 — ConvNeXt-Tiny**  
A modern CNN architecture with strong image representation capability.

**Model 4 — DINOv2 ViT-S**  
A self-supervised vision transformer.

**Model 5 — CLIP ViT-B/16**  
A vision-language pretrained model.

The final model set may be reduced if computational or implementation constraints make all five impractical.

The final selection will prioritize meaningful comparison over the number of models.

---

## 7. Feature Extraction Strategy

The primary experiment will use frozen pretrained representations.

```text
Input Image
     ↓
Pretrained Model
     ↓
Feature Vector
     ↓
Cached Feature
     ↓
Linear Classification Head
     ↓
200 Species
```

The pretrained backbone will initially remain frozen.

This substantially reduces training time because features can be extracted once and reused.

---

## 8. Linear Probe

A lightweight classifier will be trained on the frozen features.

### Initial Classifier

Linear classification head.

### Objective

Determine how useful the pretrained representation is for fine-grained bird classification without extensive model-specific fine-tuning.

### Advantages

- Fast training
- Low GPU requirements
- Reproducible
- Allows direct comparison between representations
- Avoids spending hours fine-tuning every model

---

## 9. Fine-Tuning Experiment

Full fine-tuning will **NOT** initially be performed for every model.

After the frozen-feature experiments, the most relevant 1–2 models may undergo a short fine-tuning experiment.

The purpose is to determine whether adapting the pretrained representation to CUB changes:

- Accuracy
- Background robustness
- Computational cost

The exact models and training duration will be selected after the initial results.

---

## 10. Low-Shot Experiment

A small additional experiment will investigate performance when training data is limited.

Two configurations are proposed:

- 5 images per class
- 10 images per class

The remaining test set will be kept unchanged.

The low-shot experiment will use the frozen features and lightweight classifier to keep computational cost low.

This experiment is optional and will only be performed if the main robustness experiments are completed.

---

## 11. Evaluation Metrics

### Classification

**Top-1 Accuracy**

Primary classification metric.

### Robustness

For each model:

`Accuracy Drop = Original Accuracy - Modified Condition Accuracy`

The study will report accuracy under each visual condition rather than relying only on a single robustness score.

Example:

| Model | Original | BBox | Segmented | Background Changed |
| :--- | :--- | :--- | :--- | :--- |
| ResNet-50 | — | — | — | — |
| EfficientNet | — | — | — | — |
| ConvNeXt | — | — | — | — |
| DINOv2 | — | — | — | — |
| CLIP | — | — | — | — |

---

## 12. Efficiency Metrics

The study will record:

- Number of parameters
- Feature extraction time
- Linear-head training time
- Fine-tuning time where applicable
- Inference latency
- GPU memory usage where practical

This allows model performance to be evaluated alongside computational cost.

---

## 13. Main Experimental Matrix

The primary experiment will approximately follow:

```text
             Original
                │
       ┌────────┼─────────┐
       ↓        ↓         ↓
     BBox    Segmented   BG Changed
       │        │         │
       └────────┼─────────┘
                ↓
       Multiple pretrained
       visual representations
                ↓
        Frozen feature extraction
                ↓
          Linear classifier
                ↓
          Accuracy + Robustness
                ↓
             Efficiency
```

---

## 14. Training Configuration

All models should use consistent image preprocessing wherever possible.

- **Initial Image Size:** 224 × 224
- **Dataset Split:** Official CUB train/test split.
- **Primary Training Approach:** Frozen backbone + linear classifier.
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)

### Training Philosophy

The project will prioritize:

- Reproducibility
- Controlled comparisons
- Short experiments
- Meaningful measurements

Rather than attempting large-scale hyperparameter optimization.

---

## 15. ImageNet Pretraining Caveat

The official CUB-200-2011 documentation warns that some CUB images overlap with images contained in ImageNet.

Therefore, models using ImageNet-pretrained weights may have potential pretraining/test-image overlap.

This limitation will be explicitly documented.

The study will avoid claiming that ImageNet-pretrained results constitute a completely independent evaluation.

The issue will also be discussed when interpreting differences between supervised ImageNet-pretrained models and other pretrained representations.

---

## 16. Background Robustness Protocol

The background manipulation experiments must preserve the bird itself as consistently as possible.

The segmentation masks will be used to identify foreground bird pixels.

The experiment will distinguish between:

- Removing background
- Replacing background
- Cropping around the bird

The transformation applied to each test image will be deterministic and documented.

The same transformed test set will be used across all models.

This ensures that differences in robustness are attributable to model behavior rather than different test data.

---

## 17. Research Gap

### Status

To be established through literature review and gap analysis.

Background dependence in fine-grained recognition is already an existing research topic.

Therefore, this project will **NOT** claim that:

*"Previous research has never studied background bias in CUB."*

Instead, the literature review will investigate whether there is a defensible gap in the combination of:

- Multiple pretrained model families
- Controlled background interventions
- Fine-grained CUB recognition
- Robustness measurement
- Computational efficiency
- Low-shot performance

The final research gap will be added after the literature analysis.

---

## 18. Expected Contribution

The intended contribution is a systematic empirical comparison rather than a new neural architecture.

The study aims to provide evidence about:

- Accuracy differences between pretrained visual representations.
- Changes in performance when background information is removed or modified.
- Whether certain pretraining approaches are more robust to background changes.
- The computational cost associated with different representations.
- The relationship between classification accuracy and robustness.

The exact contribution statement will be finalized after the literature review and experimental results.

---

## 19. Project Phases

### Phase 1 — Dataset and Environment

- Dataset extraction
- Segmentation verification
- PyTorch/CUDA setup
- Dataset loader

**Status:** Completed / In progress

### Phase 2 — Baseline

- Implement one pretrained model
- Extract features
- Train linear classifier
- Evaluate original CUB test images

**Goal:** Establish a working end-to-end baseline.

### Phase 3 — Model Comparison

- Add remaining selected pretrained models
- Extract and cache features
- Train identical/similar linear heads
- Compare accuracy

### Phase 4 — Robustness Evaluation

Evaluate the models on:

- Original images
- Bounding-box crops
- Segmentation-based images
- Background-modified images

Calculate accuracy changes.

### Phase 5 — Efficiency Analysis

Record:

- Parameters
- Feature extraction time
- Training time
- Inference latency
- GPU memory usage

### Phase 6 — Optional Experiments

If time permits:

- Fine-tuning of 1–2 models
- 5-shot experiment
- 10-shot experiment
- Error analysis
- Grad-CAM or related visualization

Optional experiments must not delay completion of the primary experiment.

---

## 20. Deliverables

### Code
- Dataset loader
- Segmentation processing
- Bounding-box processing
- Image transformation pipeline
- Model loading
- Feature extraction
- Feature caching
- Linear classifier training
- Evaluation
- Metric calculation

### Experimental Results
- Accuracy table
- Robustness comparison
- Accuracy-drop analysis
- Efficiency comparison
- Optional low-shot results
- Optional fine-tuning results

### Visualizations

Potential figures:
- Accuracy comparison
- Accuracy drop under background changes
- Accuracy vs computational cost
- Example original/modified images
- Optional model attention/error analysis

### Research Paper

The final paper will contain:
- Introduction
- Related Work
- Research Gap
- Methodology
- Experimental Setup
- Results
- Analysis
- Limitations
- Conclusion

---

## 21. Scope Control

The following are explicitly **not required** for the first version:

- New neural architecture
- Large-scale hyperparameter search
- Training models from scratch
- Fine-tuning every model
- Complex segmentation models
- Ensemble methods
- State-of-the-art leaderboard optimization

The primary objective is to complete a clean, reproducible benchmark.

---

## 22. Current Milestone

The immediate implementation target is:

```text
CUB-200-2011
      ↓
Pretrained Model
      ↓
Frozen Feature Extraction
      ↓
Cached Features
      ↓
Linear Classifier
      ↓
Original Test Accuracy
```

Once this works for one model, the same pipeline will be extended to the remaining selected models.

---

## 23. Future PRD Updates

### PRD v0.3

After literature/gap analysis:
- Final research gap
- Final model selection
- Final hypotheses
- Final experimental protocol

### PRD v1.0

After initial experiments:
- Final methodology
- Final experiments
- Ablation studies
- Statistical analysis where appropriate
- Final evaluation protocol

---

## 24. Success Criteria

The project is considered successful if it produces:

- A reproducible CUB-200-2011 classification pipeline.
- Results from multiple pretrained visual representations.
- Controlled evaluation under different background conditions.
- Quantitative robustness comparisons.
- Computational-efficiency measurements.
- A literature-supported research gap.
- A defensible analysis of the observed results.

The project does not require a new architecture or state-of-the-art accuracy.

---

### One thing I'd change from Claude's original plan

I **wouldn't commit to all 5 models immediately**.

Start with:

**ResNet-50 → DINOv2 ViT-S → CLIP ViT-B/16**

Those three already give you three substantially different representation/pretraining paradigms:

```text
ResNet-50   → supervised CNN
DINOv2      → self-supervised vision transformer
CLIP        → vision-language pretraining
```

If the pipeline works and time permits, add EfficientNet/ConvNeXt.