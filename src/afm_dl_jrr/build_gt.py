# Copyright (C) 2026 Jorge Rodriguez Ramos
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Ground-truth organisation and train/test splitting for AFM force-curve datasets.

Overview
--------
Provides the upstream data-preparation layer for the binary classifier in
``resnet.py``. Three utilities that transform raw instrument outputs into a
structured ``ImageFolder`` tree ready for ``torchvision.datasets.ImageFolder``:

1. Label recovery        -- ``get_ground_truth_files`` parses annotation files
2. Dataset organisation  -- ``organize_ground_truth_files`` copies files into
                            class subdirectories (``good/`` / ``bad/``)
3. Train/test partition  -- ``split_gt_for_training`` carves stratified subsets
                            from the organised ground-truth tree.

Pipeline position
-----------------
Upstream: Instrument software produces measurement files (typically ``.tdms``)
and an annotation file listing defect-free samples. This module converts those
into the directory layout expected by ``resnet.build_loaders``.

Downstream: The resulting ``train_dataset/`` and ``test_dataset/`` trees feed
``torchvision.datasets.ImageFolder`` in ``resnet.py``; class names are inferred
from subdirectory order (``bad`` = index 0, ``good`` = index 1).

Expected on-disk layout
-----------------------
Input annotation file (single-column)::

    good_samples.csv
    ───────────────────
    sample_001.tdms
    sample_005.tdms
    ...

Source directory containing raw renders::

    raw_images/
        sample_001.tif
        sample_002.tif
        ...

Output after ``organize_ground_truth_files``::

    <gt_folder>/
        good/          ← .tif files listed in annotation
        bad/           ← remaining .tif files

Output after ``split_gt_for_training``::

    <gt_folder>/
        good/              # Original source (unchanged)
        bad/               # Original source (unchanged)
        train_dataset/     # Created: training subset
            good/
            bad/
        test_dataset/      # Created: test subset
            good/
            bad/

Conventions and invariants
--------------------------
- Input files are assumed to have the extension passed to ``get_ground_truth_files``
  (default ``.tdms``); output images are always ``.tif`` (hard-coded).
- Class subdirectories must be named exactly ``good`` and ``bad`` (lower-case,
  singular). ``torchvision.datasets.ImageFolder`` sorts them alphabetically,
  yielding ``bad`` = 0, ``good`` = 1 — the same ordering expected downstream.
- ``split_gt_for_training`` uses **different ratios** for good and bad classes
  by design. Default ``bad_test_ratio=0.70`` pushes scarce negative samples into
  the test set; adjust if training-data scarcity emerges.
- Files are **copied**, not moved. Original source folders remain intact.

Known limitations
-----------------
- File-extension replacement is unanchored string substitution. An annotation
  entry like ``run.tdms_old.tdms`` becomes ``run.tif_old.tif``.
- ``organize_ground_truth_files`` silently treats any file not in the "good" list
  as "bad" without verifying it exists in the source directory.
- ``split_gt_for_training`` deletes existing ``train_dataset/`` or
  ``test_dataset/`` folders without prompting.
- Only ``.tif`` files are recognised for image operations; other extensions are
  ignored even if they exist alongside the renders.

"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import math
import random
    
def split_gt_for_training(
    gt_folder,
    good_test_ratio=0.10,
    bad_test_ratio=0.70,
    seed=42,
    train_folder_name='train_dataset',
    test_folder_name='test_dataset'
):
    """Split ground truth AFM image data into train and test subsets.

    Organizes labeled defect classification images (good/bad) into stratified
    training and test directories for supervised deep learning workflows. The
    function creates a deterministic split using a seeded random number generator
    to ensure reproducibility across runs.

    Parameters
    ----------
    gt_folder : str or Path
        Root directory containing 'good/' and 'bad/' subdirectories with source
        AFM surface quality images (.tif files).
    good_test_ratio : float, optional
        Fraction of 'good' class images to allocate to test set (0.0-1.0).
        Default is 0.10 (10% of good images).
    bad_test_ratio : float, optional
        Fraction of 'bad' class images to allocate to test set (0.0-1.0).
        Default is 0.70 (70% of bad images). Note: Higher ratio recommended for
        imbalanced datasets where bad samples are scarce.
    seed : int, optional
        Random seed for reproducible splits. Default is 42.
    train_folder_name : str, optional
        Name of the output training dataset folder. Default is 'train_dataset'.
    test_folder_name : str, optional
        Name of the output test dataset folder. Default is 'test_dataset'.

    Returns
    -------
    dict
        Dictionary containing split statistics:
        
        - 'good_total' (int): Total good images in source
        - 'bad_total' (int): Total bad images in source
        - 'train_good' (int): Good images in training set
        - 'train_bad' (int): Bad images in training set
        - 'test_good' (int): Good images in test set
        - 'test_bad' (int): Bad images in test set

    Raises
    ------
    ValueError
        If 'good' or 'bad' subdirectories are missing in gt_folder.
    ValueError
        If train/test folder names conflict with 'good' or 'bad'.

    Side Effects
    ------------
    Destructive Operations:
        - Existing train/test folders are completely removed before creation
        - Source files are COPIED, not moved (originals preserved)
        - New directory structure is populated with selected subsets

    Output Directory Structure::

        gt_folder/
            good/               # Original source (unchanged)
            bad/                # Original source (unchanged)
            train_dataset/      # Created: training subset
                good/
                bad/
            test_dataset/       # Created: test subset
                good/
                bad/

    Supported File Extensions
    -------------------------
    .tif (only, hard-coded filter)

    Notes
    -----
    - Test selection uses RANDOM SAMPLING (not sorted order)
    - Minimum 1 test image guaranteed per class if source has files and ratio > 0
    - Remaining files after test selection populate the training set
    - Files are copied with metadata preservation (shutil.copy2)
    - Reproducibility depends on consistent seed across runs

    Warnings
    --------
    Destructive: Pre-existing train/test folders matching the specified names
    will be DELETED WITHOUT PROMPT. Backup data before execution.

    Imbalanced Classes: The default bad_test_ratio (0.70) allocates most rare
    negative samples to testing—adjust if training data scarcity becomes a
    problem.
    """

    good_src = os.path.join(gt_folder, 'good')
    bad_src = os.path.join(gt_folder, 'bad')

    if not os.path.isdir(good_src) or not os.path.isdir(bad_src):
        raise ValueError(f"Expected 'good' and 'bad' folders inside: {gt_folder}")

    for folder_name in (train_folder_name, test_folder_name):
        if folder_name in ('good', 'bad'):
            raise ValueError("Train/test folder names cannot be 'good' or 'bad'")

    rng = random.Random(seed)

    def list_files(folder):
        files = [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.tif')
        ]
        files.sort()
        return files

    def pick_test(files, ratio):
        if not files or ratio <= 0:
            return []
        n_test = min(len(files), max(1, math.ceil(len(files) * ratio))) if len(files) > 0 else 0
        return rng.sample(files, n_test)

    # 1. List all files
    good_files = list_files(good_src)
    bad_files = list_files(bad_src)

    # 2. Pick the test sets
    test_good = pick_test(good_files, good_test_ratio)
    test_bad = pick_test(bad_files, bad_test_ratio)

    # 3. Everything remaining goes to the training sets
    train_good = [f for f in good_files if f not in set(test_good)]
    train_bad = [f for f in bad_files if f not in set(test_bad)]

    def reset_dir(root):
        if os.path.isdir(root):
            shutil.rmtree(root)
        elif os.path.exists(root):
            os.remove(root)
        os.makedirs(root, exist_ok=True)

    # 4. Setup directories
    train_root = os.path.join(gt_folder, train_folder_name)
    test_root = os.path.join(gt_folder, test_folder_name)

    reset_dir(train_root)
    reset_dir(test_root)

    os.makedirs(os.path.join(train_root, 'good'), exist_ok=True)
    os.makedirs(os.path.join(train_root, 'bad'), exist_ok=True)
    os.makedirs(os.path.join(test_root, 'good'), exist_ok=True)
    os.makedirs(os.path.join(test_root, 'bad'), exist_ok=True)

    # 5. Copy files
    for f in train_good:
        shutil.copy2(os.path.join(good_src, f), os.path.join(train_root, 'good', f))
    for f in train_bad:
        shutil.copy2(os.path.join(bad_src, f), os.path.join(train_root, 'bad', f))

    for f in test_good:
        shutil.copy2(os.path.join(good_src, f), os.path.join(test_root, 'good', f))
    for f in test_bad:
        shutil.copy2(os.path.join(bad_src, f), os.path.join(test_root, 'bad', f))

    # 6. Return split metrics
    return {
        "good_total": len(good_files),
        "bad_total": len(bad_files),
        "train_good": len(train_good),
        "train_bad": len(train_bad),
        "test_good": len(test_good),
        "test_bad": len(test_bad),
    }

def organize_ground_truth_files(source_folder, good_files_list):
    """
    Organizes .tif files from a source folder into good/bad and train/test ground truth directories.

    Args:
        source_folder (str): The path to the directory containing all .tif files.
        good_files_list (list): A list of filenames (with .tif extension) considered 'good'.
    """

    # Define the folder paths for ground truth data
    gt_good = './data/GT/good'
    gt_bad = './data/GT/bad'

    folders = [gt_good, gt_bad]

    # Delete if they exist, then create new folders
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # Remove the folder and all contents
        os.makedirs(folder)  # Create the folder
        print(f"Created: {folder}")

    # Get all .tif files in source directory
    tif_files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]

    print(f"\nFound {len(tif_files)} .tif files in {source_folder}")
    print(f"Comparing against {len(good_files_list)} good files identified from the CSV")

    # Convert good_files to a set for faster lookup
    good_files_set = set(good_files_list)

    for tif_file in tqdm(tif_files, desc="Organizing .tif files"):
        source_path = os.path.join(source_folder, tif_file)
        dest_path = ''

        if tif_file in good_files_set:
            dest_path = os.path.join(gt_good, tif_file)
        else:
            dest_path = os.path.join(gt_bad, tif_file)

        # Perform the copy operation
        if dest_path:
            shutil.copy2(source_path, dest_path)


def get_ground_truth_files(GT_path, File_extension: str = '.tdms'):
    """Load ground truth labels from a CSV/TXT file and map filenames to .tif.

    Reads a single-column annotation file listing defect-free samples, then
    rewrites each entry's source extension to '.tif' so the names match the
    image files produced by the preprocessing pipeline.

    Parameters
    ----------
    GT_path : str or Path
        Path to the ground truth annotation file. Delimiter is auto-detected
        (comma, tab, semicolon, whitespace) via pandas' Python engine.
    File_extension : str, optional
        Source file extension to be replaced in each filename, including the
        leading dot (e.g. '.tdms', '.spm', '.ibw'). Default is '.tdms'.

    Returns
    -------
    good_files : list of str
        Filenames from the first column of the annotation file, with
        File_extension replaced by '.tif'. Empty list if the file is missing.
    gt : bool
        True if the annotation file was found and parsed, False otherwise.
        Callers should branch on this flag rather than on list emptiness.

    Raises
    ------
    pandas.errors.ParserError (implicit)
        If the file exists but cannot be parsed with any detected delimiter.
    IndexError (implicit)
        If the file parses to a DataFrame with zero columns.

    Side Effects
    ------------
    - Prints the resolved path, record count, and first five filenames to stdout
    - Prints a "File not found" notice when GT_path is absent (non-fatal)
    - Performs NO file writes, moves, or deletions

    Notes
    -----
    - Only the FIRST column is read; additional columns are ignored silently
    - Replacement is applied to the entire string, not anchored to the suffix
    - No existence check is performed against the image directory; returned
      names may reference files that were never generated
    - Duplicate entries in the annotation file are preserved, not deduplicated
    - Absence of the annotation file is a supported state, not an error

    Warnings
    --------
    Silent Mismatch: If File_extension does not appear in the annotation
    entries (wrong case, wrong dot, or already-.tif names), the function
    returns names unchanged and reports success. Downstream classification
    will then mislabel every file as 'bad'. Verify the printed sample
    filenames before proceeding.

    Empty Extension: Passing File_extension='' causes '.tif' to be inserted
    between every character of every filename. Do not pass an empty string.

    Substring Collision: Because replacement is unanchored, an entry such as
    'run.tdms_old.tdms' becomes 'run.tif_old.tif'. Prefer passing filenames
    that carry the extension exactly once, at the end.

    Examples
    --------
    Standard TDMS workflow::

        >>> from build_gt import get_ground_truth_files
        >>> good, found = get_ground_truth_files('./data/labels.csv')
        GT file found at ./data/labels.csv
        Found 412 good files
        >>> if found:
        ...     organize_ground_truth_files('./data/raw', good)

    Non-TDMS instrument output::

        >>> good, found = get_ground_truth_files(
        ...     './data/labels.txt',
        ...     File_extension='.spm'
        ... )

    Continue without annotations::

        >>> good, found = get_ground_truth_files('./missing.csv')
        File not found
        Processing will continue without ground truth
        >>> assert good == [] and found is False

    See Also
    --------
    organize_ground_truth_files : Classify files using the returned list
    split_gt_for_training : Split the organized data into train/test subsets

    """
    good_files = []
    gt = False

    if os.path.exists(GT_path) and os.path.isfile(GT_path):
        print(f"GT file found at {GT_path}\n")
        gt = True
    else:
        print(f"File not found")
        print("Processing will continue without ground truth")
        gt = False

    if gt:
        df = pd.read_csv(GT_path, sep=None, engine='python')
        good_files = df.iloc[:,0].values
        good_files = [f.replace(File_extension, '.tif') for f in good_files] 
        print(f"Found {len(good_files)} good files")
        print(f"First 5 good files:")
        print(*good_files[:5], sep='\n')
    return good_files, gt
