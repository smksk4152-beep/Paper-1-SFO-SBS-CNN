# Paper-1-SFO-SBS-CNN
CT-based pulmonary nodule detection using CNN with hybrid SFO-SBS optimization
# Enhanced Lung Cancer Detection Using CNN with Hybrid SFO-SBS Optimization

## Project Description
This repository contains the implementation resources for CT-based pulmonary nodule detection using a Convolutional Neural Network (CNN) with hybrid Sunflower Optimization and Secretary Bird Search (SFO-SBS).

## Dataset
- Dataset: LIDC-IDRI
- Total patients: 82
- Total image samples: 13,174
- Training patients: 57
- Validation patients: 12
- Testing patients: 13

## Classification
- Class 0: Non-nodule
- Class 1: Nodule

## Feature Extraction
A total of 300 handcrafted features are extracted and reduced to 50 selected features.

## Methodology
1. CT image preprocessing
2. ROI localization
3. Handcrafted feature extraction
4. SFO-SBS feature selection
5. CNN classification
6. Performance evaluation

## Evaluation Metrics
Accuracy, sensitivity, specificity, precision, F1-score, MCC and AUC.

## Repository Contents
- preprocessing/
- feature_extraction/
- optimization/
- cnn_model/
- baselines/
- dataset_splits/
- config/
- results/

## Installation
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Reproducibility
Refer to the configuration files and execution instructions for the experimental setup.

## Data Availability
The original LIDC-IDRI images are available through the LIDC-IDRI data resource, subject to its access and usage conditions.

## Code Availability
The source code and supplementary reproducibility files are provided in this repository.
