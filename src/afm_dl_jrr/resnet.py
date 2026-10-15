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

"""Binary AFM force-curve classifier built on a partially fine-tuned ResNet-18.


Expected on-disk layout
-----------------------
    <data_dir>/
        bad/    *.tif
        good/   *.tif

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ==========================================
# FOCAL LOSS & CONFIG
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


BATCH_SIZE = 16 
EPOCHS = 40
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED, deterministic=True):
    """Set Python, NumPy and PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

# ==========================================
# GRAD-CAM ENGINE (The Interpretability Layer)
# ==========================================
class GradCAM:
    """Explains what the CV model sees in the AFM curve"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        self.handlers.append(self.target_layer.register_forward_hook(self._save_activation))
        self.handlers.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Focus on the specific class (Good vs Bad)
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations were not captured. Ensure the target layer is valid and the model has run.")

        # Weight the 512 channels of layer4 by their gradient importance
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        # Apply ReLU (we only care about features that POSITIVELY influence the class)
        cam = np.maximum(cam.detach().cpu().numpy(), 0)
        cam = cv2.resize(cam, (224, 224))

        # Normalize 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

# ==========================================
# TRANSFORMS
# ==========================================
norm_stats = ([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
# norm_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=1, translate=(0.01, 0.01)),
    transforms.ToTensor(),
    transforms.Normalize(*norm_stats)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*norm_stats)  # MUST MATCH TRAIN
])


# ==========================================
# DATA LOADERS
# ==========================================
def build_loaders(data_dir, batch_size=BATCH_SIZE, seed=None):
    full_dataset = datasets.ImageFolder(root=data_dir)
    split_seed = 42 if seed is None else seed
    rng = None
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)

    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=full_dataset.targets,
        random_state=split_seed
    )

    train_data = Subset(datasets.ImageFolder(root=data_dir, transform=train_transforms), train_idx)
    val_data = Subset(datasets.ImageFolder(root=data_dir, transform=val_transforms), val_idx)

    train_labels = np.array(full_dataset.targets)[train_idx]
    class_weights = 1. / np.bincount(train_labels)
    sampler = WeightedRandomSampler(
        weights=class_weights[train_labels].tolist(),
        num_samples=len(train_labels),
        replacement=True,
        generator=rng
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, generator=rng)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, full_dataset


# ==========================================
# MODEL BUILDER
# ==========================================
def build_model(device=DEVICE):
    model = models.resnet18(weights='IMAGENET1K_V1')
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )
    model = model.to(device)

    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-6},
        {'params': model.layer4.parameters(), 'lr': 1e-6},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], lr=1e-4)

    criterion = FocalLoss(alpha=torch.tensor([0.3, 0.7]).to(device), gamma=2.0)
    return model, optimizer, criterion


# ==========================================
# VISUALIZATION
# ==========================================
def run_visualization(img_path, model, class_names=None, device=DEVICE, show_fig=True):
    """Generates the heatmap overlay on the B&W curve."""
    model.eval()
    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = val_transforms(img_pil)
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor from val_transforms, got {type(input_tensor).__name__}")
    input_tensor = input_tensor.unsqueeze(0).to(device)

    cam_engine = GradCAM(model, model.layer4[-1])

    output = model(input_tensor)
    _, pred = torch.max(output, 1)

    heatmap = cam_engine.generate_heatmap(input_tensor, pred.item())

    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # type: ignore
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = cv2.addWeighted(heatmap_color, 0.5, img_np, 0.5, 0)

    pred_label = class_names[pred.item()] if class_names else str(pred.item())

    fig = plt.figure(figsize=(10, 5))
    # plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Curve\nTarget: {img_path.split('/')[-2]}")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Model Attention (Pred: {pred_label})")
    plt.imshow(overlay)
    plt.axis('off')
    if show_fig:
        plt.show()

    cam_engine.remove_hooks()
    return fig, overlay


# ==========================================
# TRAINING
# ==========================================
def train(model, optimizer, criterion, train_loader, val_loader, class_names,
          epochs=EPOCHS, device=DEVICE, save_path='resnet18_bw_best.pth',
          selection_metric='recall'):
    """Train the model and save the best checkpoint by recall or F1."""
    if selection_metric not in {'recall', 'f1'}:
        raise ValueError("selection_metric must be either 'recall' or 'f1'")

    best_score = 0.0
    best_fps = float('inf')
    all_preds, all_labels = [], []

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]

        good_recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0
        good_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        current_score = good_recall if selection_metric == 'recall' else good_f1

        is_better = (current_score > best_score) or (
            np.isclose(current_score, best_score) and fp < best_fps
        )
        if is_better:
            best_score = current_score
            best_fps = fp
            torch.save(model.state_dict(), save_path)

        best_marker = " ✓" if is_better else ""
        print(
            f"Epoch {epoch+1} | Recall: {good_recall:.2%} | F1: {good_f1:.2%} | "
            f"FPs: {fp} | Best by {selection_metric}: {current_score:.2%}{best_marker}"
        )

    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Load best checkpoint weights before returning
    model.load_state_dict(torch.load(save_path))

    # Calibrate threshold on validation set after training
    calibrate_precision(model, val_loader)
    return model


# ==========================================
# CALIBRATION
# ==========================================
def calibrate_precision(model, loader, device=DEVICE):
    """Print threshold calibration table using validation probabilities."""
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "Good"
            y_true.extend(labels.numpy())

    print("\n--- CALIBRATION DATA FOR THRESHOLD TUNING ---")
    print(f"Total samples in validation set: {len(loader.dataset)}")
    print(f"Class distribution in validation set: {np.bincount(y_true)} (Bad, Good)")
    print("\nUse the following data to select an optimal threshold that balances false positives and false negatives according to your operational needs.")

    print(f"\n{'Threshold':<12} | {'False Positives':<15} | {'False Negatives':<15}")
    print("-" * 50)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
        preds = [1 if p >= t else 0 for p in y_probs]
        cm = confusion_matrix(y_true, preds)
        # cm[0,1] is Bad called Good | cm[1,0] is Good called Bad
        print(f"{t:<12.2f} | {cm[0,1]:<15} | {cm[1,0]:<15}")


# ==========================================
# PREDICTION
# ==========================================
def predict_curve(model, img_path, threshold_low=0.40, threshold_high=0.60, device=DEVICE):
    """Predict one AFM curve image using the provided model instance."""
    model.eval()

    img = Image.open(img_path).convert('RGB')
    img_t = val_transforms(img)
    if not isinstance(img_t, torch.Tensor):
        raise TypeError(f"Expected a tensor from val_transforms, got {type(img_t).__name__}")
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output, dim=1)
        prob_good = probs[0][1].item()

    if prob_good >= threshold_high:
        return "GOOD", prob_good, "Automated Pass"
    if prob_good <= threshold_low:
        return "BAD", prob_good, "Automated Fail"
    return "REVIEW", prob_good, "Manual Check Required"


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    DATA_DIR = '/path/to/data/GT' 

    set_seed(SEED, deterministic=True)
    train_loader, val_loader, full_dataset = build_loaders(DATA_DIR, seed=SEED)
    model, optimizer, criterion = build_model()
    model = train(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        val_loader, 
        full_dataset.classes,
        epochs=3,
        selection_metric='f1')
