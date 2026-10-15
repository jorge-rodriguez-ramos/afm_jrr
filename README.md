# AFM Force-Curve Classifier

Binary classification system for atomic force microscopy (AFM) force curves using deep learning. Identifies defect-free ("good") vs. defective ("bad") molecular interaction traces with Grad-CAM interpretability visualizations.

> **⚠️ WARNING**: Several functions in this repository perform **destructive directory overwrites** (`organize_ground_truth_files`, `split_gt_for_training`, `make_predictions_folder`). Back up your data before execution. See [Safety Notes](#safety-notes).

---

## Features

- **ResNet-18 backbone** with partial fine-tuning (layers 3, 4, and custom classification head) for AFM force-curve images.
- **Focal Loss** ($\gamma=2.0$, $\alpha=[0.3, 0.7]$) combined with weighted random sampling to mitigate severe class imbalance.
- **Grad-CAM visualizations** providing spatial heatmaps of force-curve regions driving model predictions.
- **Stratified asymmetric train/test partitioning** designed specifically to reserve scarce negative curves for testing.
- **Threshold calibration** for operational tuning between precision and recall.
- **Colab cloud integration** supporting direct Zenodo dataset extraction and automated Google Drive backups with Paris-time timestamps.

> **Colab Walkthrough**: For an end-to-end run with public sample data, see the demo notebook: `notebooks/example_afm_dl_jrr.ipynb`.

---

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/jorge-rodriguez-ramos/afm_jrr.git
```

Requirements: Python $\ge$ 3.10, PyTorch $\ge$ 2.0, CUDA-capable GPU recommended (e.g., NVIDIA T4).

---

## Quickstart

```python
import os
from afm_dl_jrr.build_gt import get_ground_truth_files, organize_ground_truth_files, split_gt_for_training
from afm_dl_jrr.resnet import build_loaders, build_model, train

# 1. Parse annotations and structure dataset
good_files, found = get_ground_truth_files('./labels.csv', File_extension='.tdms')
if found:
    organize_ground_truth_files(source_folder='./raw_tifs', good_files_list=good_files)
    # Note: bad_test_ratio defaults to 0.70 to test against scarce bad samples
    split_gt_for_training('./data/GT', good_test_ratio=0.10, bad_test_ratio=0.70)

# 2. Build model and train
train_loader, val_loader, full_dataset = build_loaders('./data/GT/train_dataset')
model, optimizer, criterion = build_model()
model = train(
    model, optimizer, criterion, 
    train_loader, val_loader, full_dataset.classes,
    epochs=40, selection_metric='f1'
)
```

---

## File Architecture

| File | Purpose |
|------|---------|
| `build_gt.py` | Ground truth organization, label parsing, and asymmetric train/test partitioning. |
| `resnet.py` | PyTorch ResNet-18 architecture, Focal Loss, Grad-CAM generation, and training loop. |
| `aux.py` | Cloud environment utilities, batch folder/filelist inference, and Clopper-Pearson confidence interval metrics. |

---

## Known Limitations ⚠️

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Hardcoded `/content/` paths** | `aux.extract_and_organize_zip` and `aux.zip_and_save_to_gdrive` fail on local operating systems (Windows/macOS) | Use these specific functions inside Google Colab only, or run extraction/backup via standard local shell commands. |
| **Silent directory deletion** | `split_gt_for_training()` wipes existing `train_dataset/` and `test_dataset/`; `make_predictions_folder()` wipes `results/` subdirectories | Never point output paths to directories containing raw unbacked data. |
| **Unanchored string substitution** | `get_ground_truth_files()` replaces `File_extension` anywhere in the filename string | Ensure file names do not contain the target extension substring earlier in the path/stem (e.g., avoid `sample.tdms_backup.tdms`). |
| **Hardcoded output destination** | `organize_ground_truth_files()` always writes to `./data/GT/good` and `./data/GT/bad` | Ensure write permissions exist at `./data/` relative to your working directory. |
| **Strict file filter** | Only `.tif` files are recognized by data loaders and processing steps | Convert inputs (e.g., PNG/JPEG/BMP) to `.tif` before running. |

---

## Safety Notes

1. **Backup data beforehand:** Running `organize_ground_truth_files()` immediately deletes `./data/GT/good` and `./data/GT/bad` without confirmation.
2. **Verify filename alignment:** `get_ground_truth_files()` fails silently if extensions do not match case or syntax. Any sample omitted from the CSV will automatically be classified into `bad/`.
3. **Inspect default split ratios:** `split_gt_for_training()` defaults to `good_test_ratio=0.10` and `bad_test_ratio=0.70`. This heavily biases the bad class toward the evaluation set. Adjust parameters if you require balanced train/test splits.

---

## Dependencies & Licensing

### Project License
This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See [`LICENSE`](LICENSE) for details.

### Third-Party Dependencies

All dependencies are GPL-compatible permissive licenses:

| Library | License |
|---------|---------|
| PyTorch / TorchVision | BSD 3-Clause |
| SciPy | BSD 3-Clause |
| scikit-learn | BSD 3-Clause |
| NumPy | BSD 3-Clause |
| OpenCV (`opencv-python`) | Apache 2.0 |
| Matplotlib | PSF License |
| Pandas | BSD 3-Clause |
| Pillow | HPND / PIL License |
| tqdm | MPL 2.0 / MIT |
| pytz | MIT |

---

## Data Licensing

Model code is GPLv3, but training datasets and raw instrument curves are **not covered by this license**. If distributing datasets via Zenodo or other platforms, explicitly supply a data license (e.g., CC0-1.0, CC-BY-4.0) in the dataset repository.

---

## Contact & Contribution

**Author**: Jorge Rodríguez Ramos  
**Location**: Marseille, France  
**License**: GPLv3

To contribute:
1. Fork the repository.
2. Submit a pull request with a clear description of changes.
3. By submitting code, you agree to license your contribution under GPLv3.

For questions or bug reports, open an issue on GitHub.

---

## Citation

If you use this classifier in academic work, please cite:

```bibtex
@software{rodriguezramos2026afm,
  author = {Rodriguez Ramos, Jorge},
  title = {AFM Force-Curve Classifier},
  year = {2026},
  url = {https://github.com/jorge-rodriguez-ramos/afm_jrr}
}
```