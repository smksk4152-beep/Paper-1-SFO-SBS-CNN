# Repository organization

The `notebook146-organization` branch contains the modular Notebook 146 implementation and reproducibility documentation. Review the branch and compare it with `main` before merging.

Do not commit CT images, patient identifiers, generated feature arrays, model checkpoints, credentials, or restricted data.

The modular scripts start from the `.npy` matrices exported by Notebook 146. They do not yet provide a single end-to-end image-folder-to-test-metrics command. The original reference notebook is retained in the prepared local ZIP package; upload it to `notebooks/Notebook146_reference.ipynb` before considering the GitHub branch fully complete.
