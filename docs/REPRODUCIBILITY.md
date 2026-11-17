# Reproducibility notes — Notebook 146

## Reference settings
- Seed: 42; grayscale image size: 128 × 128.
- HOG: 9 orientations; 16 × 16 pixels/cell; 2 × 2 cells/block; L2-Hys.
- Initial feature matrix: 300 columns; selected subset: exactly 50.
- SFO–SBS: population 8; 10 iterations; objective is Logistic Regression validation ROC-AUC.
- Proxy classifier: LogisticRegression(max_iter=300, solver="liblinear").
- CNN input: 10 × 5 × 1; Conv2D filters 32, 64, 128; dense units 128 and 64; dropout 0.50.
- Adam learning rate 0.001; binary cross-entropy; batch 32; up to 30 epochs.
- Checkpoint/early stopping monitor: validation AUC; patience 7.

## Search update probabilities
SFO copy-to-best probability: `0.20 * (1 - iteration / max_iterations)`.
SBS bit-flip probability: `0.10 + 0.20 * (iteration / max_iterations)`.
Masks are repaired to contain exactly 50 selected features.

## Important cautions
1. HOG alone exceeds 300 values with the configured image/HOG settings. Since the concatenated vector is truncated to its first 300 values, the effective matrix is HOG-only. This preserves Notebook 146 behavior.
2. Fit StandardScaler on training only.
3. A fresh stochastic SFO–SBS run may select a different subset. Preserve each run's seed, indices, and convergence history.
4. Notebook 146 does not optimize CNN hyperparameters; CNN settings are fixed.
5. Keep test data out of feature selection, threshold selection, and model selection.
6. Do not describe nodule/non-nodule classification as benign/malignant classification or a definitive cancer diagnosis.
