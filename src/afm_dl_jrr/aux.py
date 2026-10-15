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


"""Auxiliary utilities for AFM force-curve classification workflows.

Overview
--------
Glue code connecting the core classifier (``resnet.py``) to data management,
batch inference, and evaluation. Utilities span three domains:

1. Cloud plumbing            -- ``extract_and_organize_zip``,
                                ``zip_and_save_to_gdrive``
2. Batch inference           -- ``make_predictions_filelist``,
                                ``make_predictions_folder``
3. Evaluation & statistics   -- ``evaluate_predictions``

"""
import zipfile
import os
import shutil # Import shutil for moving and removing directories

def extract_and_organize_zip(zip_path, final_target_dir):
    """Extract a ZIP archive and organize its contents into a target directory.

    If the ZIP contains a single top-level folder, its contents are moved
    directly into ``final_target_dir`` (flattening one level). Otherwise all
    extracted items are moved as-is. A temporary directory is used during
    extraction and removed afterward.

    Args:
        zip_path (str): Path to the ZIP file to extract.
        final_target_dir (str): Destination directory for the extracted contents.
    """
    print(f"📦 Extracting contents of {zip_path}...")

    # Warn and ask for confirmation if target directory already exists
    if os.path.exists(final_target_dir):
        print(f"⚠️  Warning: '{final_target_dir}' already exists and will be deleted.")
        response = input("Do you want to proceed? [y/N]: ").strip().lower()
        if response != 'y':
            print("❌ Operation cancelled.")
            return
        shutil.rmtree(final_target_dir)
        print(f"🗑️  Deleted '{final_target_dir}'.")

    # Create a temporary directory for extraction to handle potential top-level folder in zip
    temp_extract_dir = '/content/temp_zip_extract'
    os.makedirs(temp_extract_dir, exist_ok=True)
    os.makedirs(final_target_dir, exist_ok=True) # Ensure the final target directory exists

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # Find the top-level directory within the temporary extraction location
    extracted_items = os.listdir(temp_extract_dir)

    if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract_dir, extracted_items[0])):
        # If there's a single top-level directory (e.g., 'GT2'), move its contents up
        source_content_dir = os.path.join(temp_extract_dir, extracted_items[0])
        print(f"Moving contents from '{source_content_dir}' to '{final_target_dir}'")
        for item in os.listdir(source_content_dir):
            shutil.move(os.path.join(source_content_dir, item), final_target_dir)
        shutil.rmtree(temp_extract_dir) # Remove the temp directory after moving contents
        print(f"Removed temporary extraction directory '{temp_extract_dir}'")
    else:
        # If no single top-level directory, or multiple items, move all extracted contents directly
        print(f"Moving extracted contents from '{temp_extract_dir}' to '{final_target_dir}'")
        for item in extracted_items:
            shutil.move(os.path.join(temp_extract_dir, item), final_target_dir)
        shutil.rmtree(temp_extract_dir)
        print(f"Removed temporary extraction directory '{temp_extract_dir}'")

    print("✅ Extraction and organization complete!")

import os
import zipfile
from datetime import datetime
import pytz # Import pytz for timezone handling
import shutil # Import shutil for make_archive

def zip_and_save_to_gdrive(source_path: str, google_drive_path: str):
    """
    Zips a specified file or folder and saves it to a Google Drive path.

    The zipped content will include a timestamp based on the 'Europe/Paris' timezone.
    The name of the zip file will be derived from the source_path.

    Args:
        source_path (str): The full path to the file or directory to be zipped.
        google_drive_path (str): The full Google Drive path where the zipped content
                                 should be saved (e.g., '/content/drive/MyDrive/backups').

    Usage:
        # zip and save a model file and a folder to Google Drive
        model_file = '/content/resnet18_bw_best.pth'
        drive_save_location = input("Enter the full Google Drive path (e.g., /content/drive/MyDrive/models): ")
        zip_and_save_to_gdrive(model_file, drive_save_location)

        some_folder = '/content/GT' # Assuming GT folder exists after previous steps
        drive_save_location_folder = input("Enter the full Google Drive path to save the folder (e.g., /content/drive/MyDrive/data_backups): ")
        zip_and_save_to_gdrive(some_folder, drive_save_location_folder)                                 
    """
    if not os.path.exists(source_path):
        print(f"Error: Source path not found at {source_path}")
        return

    # Define the Paris timezone
    paris_timezone = pytz.timezone('Europe/Paris')
    # Get the current time in Paris timezone
    now_in_paris = datetime.now(paris_timezone)
    # Format the timestamp for unique filename
    timestamp = now_in_paris.strftime("%Y%m%d_%H%M%S")

    # Determine base name for the zip file (e.g., 'resnet18_bw_best.pth' or 'GT')
    base_name = os.path.basename(source_path)

    # Define the full name for the temporary zipped file
    temp_zip_full_name = f'{base_name}_{timestamp}.zip'
    temp_zip_path = '' # Initialize outside try-block for cleanup

    # Ensure the Google Drive path exists
    if not os.path.exists(google_drive_path):
        print(f"Creating directory: {google_drive_path}")
        os.makedirs(google_drive_path, exist_ok=True)

    try:
        if os.path.isfile(source_path):
            temp_zip_path = os.path.join('/content', temp_zip_full_name)
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add the file to the root of the zip with its original name
                zipf.write(source_path, base_name)
        elif os.path.isdir(source_path):
            # For a directory, shutil.make_archive is simpler.
            # It creates an archive at 'base_name' prefix, in 'root_dir', of 'base_dir'
            # The returned value is the full path to the archive.
            # The archive name will be `base_name_timestamp.zip`
            archive_base_name_for_shutil = os.path.splitext(temp_zip_full_name)[0] # remove .zip extension
            temp_zip_path = shutil.make_archive(
                os.path.join('/content', archive_base_name_for_shutil), # full path to the archive prefix
                'zip',
                root_dir=os.path.dirname(source_path), # Start archiving from this directory
                base_dir=os.path.basename(source_path)  # This is the directory we want to archive within root_dir
            )
        else:
            print(f"Error: Unsupported source type at {source_path}. Must be a file or a directory.")
            return

        # Move the zipped file to the specified Google Drive path
        final_save_path = os.path.join(google_drive_path, os.path.basename(temp_zip_path))
        shutil.move(temp_zip_path, final_save_path) # Use shutil.move for cross-device links
        print(f"✅ Content successfully zipped and saved to: {final_save_path}")

    except Exception as e:
        print(f"Error saving content to Google Drive: {e}")
        if os.path.exists(temp_zip_path): # Check if temp_zip_path was created before printing its path
            print(f"The zipped content is temporarily available at: {temp_zip_path}")

    # Clean up the temporary zip file if it wasn't moved (e.g., if an error occurred in shutil.move's cleanup)
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)

def make_predictions_filelist(model, file_list, threshold_low=0.5, threshold_high=0.50):
    """
    Processes images in file_list, visualizations are not saved
    and returns a summary DataFrame.
    """
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    import pandas as pd
    import shutil
    from pathlib import Path
    from .resnet import predict_curve, run_visualization
    n_good = 0
    n_bad = 0
    n_verify = 0
    prediction_results = []

    # Iterate through files
    for file in file_list:
        if file.endswith('.tif'):
            label, conf, action = predict_curve(
                model=model,
                img_path=file,
                threshold_low=threshold_low,
                threshold_high=threshold_high
            )

            prediction_results.append({"filename": file, "label": label, "score": conf})

            # Visualization logic
            show_fig = (label != 'BAD')
            if show_fig:
                print(f"{file} --> {label} \n\t Confidence Good: {conf:.2%} | Action: {action}\n")
            
            fig, _ = run_visualization(file, model, show_fig=show_fig)

            file_jpg = file.replace('.tif', '.jpg')
            target_folder = label.lower() if label != 'REVIEW' else 'verify'
            
            if label == 'GOOD': n_good += 1
            elif label == 'BAD': n_bad += 1
            else: n_verify += 1

            
            plt.close(fig)

    print(f"Summary: Good: {n_good}, To check: {n_verify}, Bad: {n_bad}")
    
    df = pd.DataFrame(prediction_results)
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    return df, n_good, n_bad, n_verify

def make_predictions_folder(model, work_dir, threshold_low=0.5, threshold_high=0.60):
    """
    Processes images in work_dir, saves visualizations to subfolders,
    and returns a summary DataFrame.
    """
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    import pandas as pd
    import shutil
    from pathlib import Path
    from .resnet import predict_curve, run_visualization
    n_good = 0
    n_bad = 0
    n_verify = 0
    prediction_results = []

    # Setup results directories
    base_results_path = os.path.join(work_dir, 'results')
    for category in ['good', 'verify', 'bad']:
        cat_path = os.path.join(base_results_path, category)
        if os.path.exists(cat_path):
            shutil.rmtree(cat_path)
        os.makedirs(cat_path)

    # Iterate through files
    for root, dirs, files in os.walk(work_dir):
        # Skip the results folder itself to avoid infinite loops/double processing
        if 'results' in root: continue
        
        for file in files:
            if file.endswith('.tif'):
                img_path = os.path.join(root, file)
                label, conf, action = predict_curve(
                    model=model,
                    img_path=img_path,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high
                )

                prediction_results.append({"filename": file, "label": label, "score": conf})

                # Visualization logic
                show_fig = (label != 'BAD')
                if show_fig:
                    print(f"{file} --> {label} \n\t Confidence Good: {conf:.2%} | Action: {action}\n")
                
                fig, _ = run_visualization(img_path, model, show_fig=show_fig)

                file_jpg = file.replace('.tif', '.jpg')
                target_folder = label.lower() if label != 'REVIEW' else 'verify'
                
                if label == 'GOOD': n_good += 1
                elif label == 'BAD': n_bad += 1
                else: n_verify += 1

                fig.savefig(
                    os.path.join(base_results_path, target_folder, file_jpg),
                    bbox_inches='tight',
                    pad_inches=0
                )
                plt.close(fig)

    print(f"Summary: Good: {n_good}, To check: {n_verify}, Bad: {n_bad}")
    
    df = pd.DataFrame(prediction_results)
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    return df, n_good, n_bad, n_verify

def evaluate_predictions(pred_df, good_files, Extension_filename: str = '.tdms'):
    """
    Evaluate model predictions with ground truth, compute statistics,
    and plot Clopper-Pearson 95% confidence intervals.
    
    Parameters:
    -----------
    pred_df : pd.DataFrame
        Predictions dataframe with 'filename' and 'label' columns
    good_files : list
        List of ground truth "good" filenames
        Note: prediction label "REVIEW" is treated as "GOOD" for stats.
        
    Returns:
    --------
    pred_eval : pd.DataFrame
        Evaluation dataframe with predictions, ground truth, and correctness flags
    """
    from scipy.stats import binom
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Build dataframe with prediction + ground truth from good_files
    pred_eval = pred_df.copy()
    # Change label REVIEW to GOOD
    pred_eval["label"] = pred_eval["label"].astype(str).str.upper().replace({"REVIEW": "GOOD"})
    
    # Resolve filename column robustly
    filename_candidates = ["filename", "filenames", "file", "name"]
    filename_col = next((c for c in filename_candidates if c in pred_eval.columns), None)
    if filename_col is None:
        filename_col = pred_eval.columns[0]
    
    # Keep only filename + predicted label
    pred_eval = pred_eval[[filename_col, "label"]].copy()
    pred_eval = pred_eval.rename(columns={filename_col: "filename", "label": "prediction"})
    
    # Normalize filename extension to .tif
    pred_eval["filename"] = pred_eval["filename"].astype(str).str.replace(Extension_filename, ".tif", regex=False)
    
    # Ground truth from good_files list
    good_set = set(good_files)
    pred_eval["ground_truth"] = pred_eval["filename"].isin(good_set).map({True: "GOOD", False: "BAD"})
    # Treat VERIFY predictions as GOOD for all evaluation statistics.
    pred_for_stats = pred_eval["prediction"].astype(str).str.upper().replace({"VERIFY": "GOOD"})
    pred_eval["is_correct"] = pred_for_stats.eq(pred_eval["ground_truth"])
    
    # === Compute metrics ===
    y_true = pred_eval["ground_truth"].astype(str).str.upper()
    y_pred = pred_for_stats
    
    print(f"Total files evaluated: {len(pred_eval)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=["GOOD", "BAD"]),
        index=["GOOD", "BAD"],
        columns=["GOOD", "BAD"]
    ))
    
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=["GOOD", "BAD"]))
    
    # === Clopper-Pearson CIs ===
    def clopper_pearson_ci(successes, n, alpha=0.05):
        """Compute Clopper-Pearson binomial confidence interval."""
        if n == 0:
            return None, None
        ci_lower = binom.ppf(alpha/2, n, successes/n) / n
        ci_upper = binom.ppf(1 - alpha/2, n, successes/n) / n
        return ci_lower, ci_upper
    
    alpha = 0.05
    
    # Overall accuracy
    n_total = len(pred_eval)
    successes_total = pred_eval['is_correct'].sum()
    ci_lower, ci_upper = clopper_pearson_ci(successes_total, n_total, alpha)
    print(f"OVERALL:")
    print(f"  Total: {n_total}, Correct: {successes_total}, Accuracy: {successes_total/n_total:.4f}")
    print(f"  Clopper-Pearson 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
    
    # GOOD files
    good_mask = pred_eval['ground_truth'] == 'GOOD'
    n_good = good_mask.sum()
    successes_good = pred_eval[good_mask]['is_correct'].sum()
    ci_lower_good, ci_upper_good = clopper_pearson_ci(successes_good, n_good, alpha)
    print(f"GOOD FILES:")
    print(f"  Total: {n_good}, Correct: {successes_good}, TPR (Recall): {successes_good/n_good:.4f}")
    print(f"  Clopper-Pearson 95% CI: [{ci_lower_good:.4f}, {ci_upper_good:.4f}]\n")
    
    # BAD files
    bad_mask = pred_eval['ground_truth'] == 'BAD'
    n_bad = bad_mask.sum()
    successes_bad = pred_eval[bad_mask]['is_correct'].sum()
    ci_lower_bad, ci_upper_bad = clopper_pearson_ci(successes_bad, n_bad, alpha)
    print(f"BAD FILES:")
    print(f"  Total: {n_bad}, Correct: {successes_bad}, TNR (Specificity): {successes_bad/n_bad:.4f}")
    print(f"  Clopper-Pearson 95% CI: [{ci_lower_bad:.4f}, {ci_upper_bad:.4f}]")
    
    # === Plot ===
    categories = ['Accuracy', 'TPR (Recall)', 'TNR (Specificity)']
    scores = [
        successes_total / n_total,
        successes_good / n_good if n_good > 0 else 0,
        successes_bad / n_bad if n_bad > 0 else 0
    ]
    ci_lowers = [ci_lower, ci_lower_good, ci_lower_bad]
    ci_uppers = [ci_upper, ci_upper_good, ci_upper_bad]
    
    # Calculate error bar lengths
    errors_lower = np.array(scores) - np.array(ci_lowers)
    errors_upper = np.array(ci_uppers) - np.array(scores)
    
    # Create plot with dot and error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(categories))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # Plot each errorbar individually with its own color
    for i, (x, y, color) in enumerate(zip(x_pos, scores, colors)):
        ax.errorbar(x, y, yerr=[[errors_lower[i]], [errors_upper[i]]], 
                    fmt='o', markersize=12, capsize=8, capthick=2, 
                    elinewidth=2, color=color, ecolor=color, alpha=0.7)
    
    # Add dashed line at 0.5
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, linewidth=1.5)
    
    # Customize plot
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Clopper-Pearson 95% Confidence Intervals by Metric', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylim((0, 1.05))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (cat, score) in enumerate(zip(categories, scores)):
        ax.text(i + 0.05, score + 0.02, f'{score:.3f}', ha='left', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return pred_eval