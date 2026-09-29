# PRD v0.1 — Bird Species Recognition

## 1. Project Overview

### Project Title
Bird Species Recognition using Deep Learning

### Dataset
Caltech-UCSD Birds-200-2011 (CUB-200-2011)

### Project Objective
Develop a deep learning system capable of classifying bird images into their corresponding species.

The immediate objective is to establish a reproducible baseline classification system. The final research contribution will be determined after literature review and gap analysis.

---

## 2. Problem Statement

Fine-grained bird species classification is challenging because visually similar species can have subtle differences in:

- Shape
- Beak structure
- Wing characteristics
- Feather patterns
- Colour
- Body structure

The same species may also appear significantly different because of:

- Pose
- Viewpoint
- Lighting
- Background
- Image conditions

The system should classify an input bird image into one of the 200 species represented in CUB-200-2011.

---

## 3. Dataset

### CUB-200-2011

CUB-200-2011 contains:

- 11,788 images
- 200 bird species
- Training/test split
- Bounding-box annotations
- 15 body-part annotations
- 312 visual attributes

### Dataset Structure

```text
CUB_200_2011/
├── attributes/
├── images/
├── parts/
├── bounding_boxes.txt
├── classes.txt
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── README
```

The provided `train_test_split.txt` will be used to maintain the official dataset split.

---

## 4. Initial Project Scope

The first implementation will focus on standard image classification.

### Initial Pipeline

```text
Bird Image
    ↓
Image Preprocessing
    ↓
CNN Feature Extraction
    ↓
Classification Layer
    ↓
200 Bird Species
```

The first version will **NOT** attempt to implement:

- Body-part feature fusion
- Attribute prediction
- Attention mechanisms
- Multi-task learning
- Ensemble models
- Custom localization networks
- Complex segmentation
- Novel architectures

These may be investigated later after the literature and gap analysis.

---

## 5. Baseline Model

### Candidate Model

**ResNet-18**

ResNet-18 is selected as the initial baseline because it is:

- Relatively lightweight
- Well established
- Suitable for transfer learning
- Practical on the available RTX 4050 Laptop GPU
- Fast enough for rapid experimentation
- Simple enough to reproduce and analyze

**Input**
224 × 224 RGB image

**Output**
200-class probability distribution

---

## 6. Pretraining Consideration

The official CUB-200-2011 documentation contains the following warning:

> Images in this dataset overlap with images in ImageNet. Exercise caution when using networks pretrained with ImageNet.

This creates a potential test-set overlap issue when evaluating ImageNet-pretrained models.

Therefore, ImageNet-pretrained and non-pretrained models should be treated as distinct experimental conditions rather than assuming that ImageNet transfer learning provides a completely independent evaluation.

### Initial Approach

The first baseline will investigate:

- ResNet-18 trained with transfer learning

If computational time permits, a second reference experiment will be performed:

- ResNet-18 trained from scratch

This allows the results of transfer learning to be compared against a model that does not use ImageNet pretraining.

The potential ImageNet/CUB overlap will be documented as an evaluation limitation in the final research paper.

---

## 7. Data Preprocessing

### Training

Images will initially undergo:

- Resize/crop to 224 × 224
- Random horizontal flip
- Random resized crop
- Small random rotation
- Moderate colour/brightness augmentation
- ImageNet normalization where applicable

### Evaluation

Test images will use deterministic preprocessing:

- Resize
- Center crop
- Normalization

No random augmentation will be applied during evaluation.

---

## 8. Training Configuration

The initial configuration will be kept intentionally simple.

| Parameter | Initial Setting |
| :--- | :--- |
| Architecture | ResNet-18 |
| Input Size | 224 × 224 |
| Classes | 200 |
| Batch Size | 32 initially |
| Initial Epochs | 10 |
| Loss | Cross-Entropy Loss |
| Optimizer | AdamW |
| GPU | NVIDIA RTX 4050 6 GB |
| Mixed Precision | Considered for faster training |

These are initial settings, not final experimental conclusions.

A short pilot run will be performed before committing to a longer training run.

---

## 9. Training Strategy

The project will prioritize rapid experimentation because the initial development period is limited.

### Stage 1 — Pipeline Verification

Verify that:

- Dataset loads correctly
- Labels are mapped correctly
- Train/test split is correct
- GPU is being used
- Images are processed correctly
- Model produces 200-class outputs

### Stage 2 — Short Pilot

Run a short training experiment to verify:

- Loss decreases
- Accuracy improves
- Training is stable
- GPU memory usage is acceptable
- Training speed is practical

### Stage 3 — Baseline Training

After the pilot succeeds, perform the baseline training run and save the best model checkpoint.

---

## 10. Evaluation

### Primary Metric

**Top-1 Classification Accuracy**

The percentage of test images for which the predicted bird species matches the ground-truth species.

### Additional Measurements

The experiment should record:

- Training loss
- Test/validation loss
- Training accuracy
- Test accuracy
- Training time
- GPU memory usage where useful

### Baseline Result

The baseline should produce a record similar to:

- **Model:**
- **Pretraining:**
- **Input Resolution:**
- **Batch Size:**
- **Epochs:**
- **Optimizer:**
- **Training Time:**
- **Best Training Accuracy:**
- **Test Accuracy:**

---

## 11. Baseline Experiment

The first completed experiment should answer:

*How well can a standard ResNet-18 image-classification model classify the 200 bird species in CUB-200-2011?*

The baseline will serve as the reference point for all future experiments.

Future improvements should be evaluated against this baseline rather than reported without a comparison.

---

## 12. Reproducibility

The implementation should record:

- Python version
- PyTorch version
- Torchvision version
- GPU
- Model configuration
- Dataset split
- Image resolution
- Batch size
- Learning rate
- Number of epochs
- Random seed where applicable

The trained model checkpoint and relevant training results should be preserved.

---

## 13. Current Deliverable

The current project milestone is complete when the following are available:

- **Code**
  - Dataset loader
  - Data preprocessing
  - Data augmentation
  - ResNet-18 model
  - Training loop
  - Evaluation loop
  - Checkpoint saving
- **Results**
  - Training loss curve
  - Accuracy measurements
  - Final baseline test accuracy
  - Training time
- **Documentation**
  - Model configuration
  - Experimental settings
  - Dataset information
  - ImageNet overlap limitation

---

## 14. Research Gap

Not defined yet.

The research gap will be established through a separate literature review and gap analysis.

The gap analysis will determine:

- Limitations of existing approaches
- Potential research direction
- Proposed improvement
- Research question
- Additional experiments
- Final methodology

No unsupported research gap or novelty claim will be included in this version of the PRD.

---

## 15. Future PRD Updates

This document represents PRD v0.1 and intentionally covers only the baseline stage.

After the literature/gap analysis, the PRD can be extended.

### PRD v0.2

Will incorporate the identified research gap and define the proposed approach.

### PRD v1.0

Will define the finalized:

- Research question
- Proposed methodology
- Experiments
- Ablation studies
- Error analysis
- Evaluation strategy
- Research contribution

The existing baseline will remain the reference experiment for comparison.
