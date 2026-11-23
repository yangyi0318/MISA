import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

CIFAR100_MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)




def normalize_image(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if dataset_name == "cifar100":
        mean = CIFAR100_MEAN.to(x.device, x.dtype)
        std = CIFAR100_STD.to(x.device, x.dtype)
    else:
        mean = CIFAR10_MEAN.to(x.device, x.dtype)
        std = CIFAR10_STD.to(x.device, x.dtype)

    return (x - mean) / std


def make_grad_scaler(device: torch.device, enabled: bool) -> GradScaler:
    amp_enabled = enabled and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return GradScaler(device=device_type, enabled=amp_enabled)


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n


# -----------------------------
# Model definitions
# -----------------------------

def make_resnet18(dataset_name: str) -> nn.Module:
    model = models.resnet18(weights=None)

    if dataset_name in ["cifar10", "cifar100"]:
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    model.fc = nn.Identity()
    return model


class StandardSurrogate(nn.Module):
    def __init__(self, num_classes: int = 10, dataset_name: str = "cifar10") -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.backbone = make_resnet18(dataset_name)
        self.fc = nn.Linear(512, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(normalize_image(x, self.dataset_name))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encode(x)
        logits = self.fc(feat)
        return {"feat": feat, "logits": logits}


def get_model_logits(output):
    return output["logits"] if isinstance(output, dict) else output


class VictimNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            dataset_name: str = "cifar10",
            model_name: str = "resnet18",
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.model_name = model_name

        if model_name == "resnet18":
            self.backbone = make_resnet18(dataset_name)
            self.fc = nn.Linear(512, num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        print(f"Victim Net is {self.model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == "resnet18":
            feat = self.backbone(normalize_image(x, self.dataset_name))
            return self.fc(feat)

        return self.backbone(normalize_image(x, self.dataset_name))


# -----------------------------
# Losses
# -----------------------------


def target_margin_loss(logits: torch.Tensor, target_label: int) -> torch.Tensor:
    target_logit = logits[:, target_label]

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, target_label] = True

    min_value = torch.finfo(logits.dtype).min
    other_logits = logits.masked_fill(mask, min_value)
    other_max = other_logits.max(dim=1).values

    margin = target_logit - other_max
    return -margin.mean()

# -----------------------------
# Dataset wrappers
# -----------------------------

class DatasetWithTransform(Dataset):
    def __init__(
            self,
            base_ds: Dataset,
            transform=None,
    ) -> None:
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x, y = self.base_ds[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class PoisonedDataset(Dataset):
    def __init__(
            self,
            clean_ds: Dataset,
            poison_indices: torch.Tensor,
            poisoned_images: torch.Tensor,
    ) -> None:
        self.clean_ds = clean_ds
        self.poison_map = {
            int(idx.item()): img
            for idx, img in zip(poison_indices, poisoned_images)
        }

    def __len__(self) -> int:
        return len(self.clean_ds)

    def __getitem__(self, idx: int):
        if idx in self.poison_map:
            _, y = self.clean_ds[idx]
            return self.poison_map[idx], y

        return self.clean_ds[idx]




def load_dataset_and_labels(
        root: str,
        train: bool,
        download: bool,
        dataset_name: str = "cifar10",
) -> Tuple[Dataset, torch.Tensor]:


    ds_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    ds = ds_class(
        root=root,
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )

    if hasattr(ds, "targets"):
        labels = torch.tensor(ds.targets)
    elif hasattr(ds, "labels"):
        labels = torch.tensor(ds.labels)
    else:
        labels = torch.tensor([y for _, y in ds])

    return ds, labels


def build_surrogate_train_dataset(
        root: str,
        dataset_name: str,
        download: bool,
) -> Dataset:

    ds_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    return ds_class(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    )


def get_dataset_info(dataset_name: str) -> Tuple[int, int]:
    if dataset_name == "cifar100":
        return 100, 32
    return 10, 32


def get_train_transform(dataset_name: str):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])


# -----------------------------
# Evaluation helpers
# -----------------------------

def evaluate_classifier(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = get_model_logits(model(x))
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / max(total, 1)


def evaluate_classifier_loss_and_acc(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


# -----------------------------
# Surrogate training
# -----------------------------

def train_surrogate_standard(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        epochs: int,
        lr: float,
        weight_decay: float,
        amp: bool,
) -> None:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    scaler = make_grad_scaler(device, amp)
    amp_enabled = amp and device.type == "cuda"

    for epoch in range(1, epochs + 1):
        model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x)
                logits = out["logits"]
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))

        scheduler.step()

        test_acc = evaluate_classifier(model, test_loader, device)

        print(
            f"[Standard Surrogate][Epoch {epoch:03d}/{epochs}] "
            f"train_loss={loss_meter.avg:.4f} "
            f"train_acc={acc_meter.avg * 100:.2f}% "
            f"test_acc={test_acc * 100:.2f}%"
        )


@dataclass
class TriggerState:
    delta_shared: torch.Tensor
    eps: float
    target_label: int


def optimize_trigger(
    surrogate: nn.Module,
    train_dataset: Dataset,
    train_labels: torch.Tensor,
    target_label: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr_delta: float,
    eps: float,
    amp: bool,
    img_size: int
) -> TriggerState:
    surrogate.eval()

    for p in surrogate.parameters():
        p.requires_grad_(False)

    delta = nn.Parameter(torch.zeros(1, 3, img_size, img_size, device=device))

    optimizer = torch.optim.Adam(
        [{"params": [delta], "lr": lr_delta}]
    )


    scaler = make_grad_scaler(device, amp)
    amp_enabled = amp and device.type == "cuda"

    target_idx = (train_labels == target_label).nonzero(as_tuple=False).squeeze(1)

    if target_idx.numel() == 0:
        raise ValueError(f"No training samples found for target_label={target_label}")

    target_ds = torch.utils.data.Subset(train_dataset, target_idx.tolist())
    safe_batch_size = min(batch_size, len(target_ds))

    if safe_batch_size <= 0:
        raise ValueError(
            f"Invalid trigger batch size. target_ds size={len(target_ds)}, "
            f"requested batch_size={batch_size}"
        )

    target_loader = DataLoader(
        target_ds,
        batch_size=safe_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )
    loader_iter = iter(target_loader)

    def compute_hard_target_margin(logits: torch.Tensor, target_label: int) -> torch.Tensor:
        """
        Hard target margin:
            M_t(x) = z_t(x) - max_{k != t} z_k(x)

        Returns:
            margins: Tensor of shape [B]
        """
        target_logits = logits[:, target_label]

        other_logits = logits.clone()
        other_logits[:, target_label] = -float("inf")
        max_other_logits = other_logits.max(dim=1).values

        margins = target_logits - max_other_logits
        return margins

    from time import time
    t1 = time()
    for step in range(1, steps + 1):
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(target_loader)
            x, _ = next(loader_iter)

        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            x_trig = torch.clamp(x + delta, 0.0, 1.0)

            out_trig = surrogate(x_trig)
            logits = out_trig["logits"]

            loss = target_margin_loss(logits, target_label)

            margins = compute_hard_target_margin(logits, target_label)
            margin_mean = margins.mean()
            margin_min = margins.min()
            margin_max = margins.max()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            delta.clamp_(-eps, eps)

        if step == 1 or step % 50 == 0 or step == steps:
            print(
                f"[Trigger][Step {step:04d}/{steps}] "
                f"loss={loss.item():.4f} "
                f"margin_mean={margin_mean.item():.4f} "
                f"margin_min={margin_min.item():.4f} "
                f"margin_max={margin_max.item():.4f}"
            )
    t2 = time()
    print(f"trigger optimization took {t2 - t1} seconds")

    return TriggerState(
        delta_shared=delta.detach().cpu(),
        eps=eps,
        target_label=target_label,
    )


# -----------------------------
# Trigger application helpers
# -----------------------------

@torch.no_grad()
def make_triggered_batch(
        x: torch.Tensor,
        delta_shared: torch.Tensor,
        amplification: float,
) -> torch.Tensor:
    delta_shared = delta_shared.to(x.device, dtype=x.dtype)
    delta = amplification * delta_shared
    return torch.clamp(x + delta, 0.0, 1.0)


# -----------------------------
# Poison dataset creation
# -----------------------------

@torch.no_grad()
def build_poisoned_training_set(
        train_dataset: Dataset,
        poison_indices: torch.Tensor,
        delta_shared: torch.Tensor,
        batch_size: int,
        device: torch.device,
) -> PoisonedDataset:
    if poison_indices.numel() == 0:
        raise ValueError("poison_indices is empty.")

    subset = torch.utils.data.Subset(train_dataset, poison_indices.tolist())

    loader = DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
        num_workers=2,
    )

    poisoned_tensors = []

    for x, _ in loader:
        x_bd = make_triggered_batch(
            x=x.to(device, non_blocking=True),
            delta_shared=delta_shared,
            amplification=1.0,
        )
        poisoned_tensors.append(x_bd.cpu())

    subset_poisoned = torch.cat(poisoned_tensors, dim=0)

    poisoned_dataset = PoisonedDataset(
        clean_ds=train_dataset,
        poison_indices=poison_indices,
        poisoned_images=subset_poisoned,
    )

    return poisoned_dataset

@torch.no_grad()
def save_poisoned_images(
        train_dataset: Dataset,
        poison_indices: torch.Tensor,
        delta_shared: torch.Tensor,
        trigger_amplifications: List[float],
        save_dir: str,
        max_save: int = -1,
        grid_count: int = 32,
        test_dataset: Optional[Dataset] = None,
        test_labels: Optional[torch.Tensor] = None,
        test_save_ratio_per_class: float = 0.10,
        test_sample_seed: int = 42,
) -> None:
    sample_dir = os.path.join(save_dir, "poisoned_samples")
    clean_dir = os.path.join(sample_dir, "clean")
    poisoned_root_dir = os.path.join(sample_dir, "poisoned")
    residual_signed_root_dir = os.path.join(sample_dir, "residual_signed")
    residual_abs_root_dir = os.path.join(sample_dir, "residual_abs")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(poisoned_root_dir, exist_ok=True)
    os.makedirs(residual_signed_root_dir, exist_ok=True)
    os.makedirs(residual_abs_root_dir, exist_ok=True)

    def amp_to_name(amp: float) -> str:
        # e.g. 1.0 -> 1p0, 2.5 -> 2p5
        return str(float(amp)).replace(".", "p").replace("-", "m")

    def label_to_name(label: int) -> str:
        return f"class_{int(label):03d}"

    def tensorize_image(x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = transforms.ToTensor()(x)
        return x.detach().cpu().clamp(0.0, 1.0)

    def save_residual_visualizations(
            clean_x: torch.Tensor,
            poisoned_x: torch.Tensor,
            signed_path: str,
            abs_path: str,
    ) -> None:
        residual = poisoned_x - clean_x
        max_abs = residual.abs().max()

        if max_abs.item() < 1e-12:
            residual_signed_vis = torch.full_like(residual, 0.5)
            residual_abs_vis = torch.zeros_like(residual)
        else:
            residual_signed_vis = residual / (2.0 * max_abs) + 0.5
            residual_signed_vis = residual_signed_vis.clamp(0.0, 1.0)

            residual_abs_vis = residual.abs() / max_abs
            residual_abs_vis = residual_abs_vis.clamp(0.0, 1.0)

        save_image(residual_signed_vis, signed_path)
        save_image(residual_abs_vis, abs_path)

    # Remove duplicates while preserving order.
    amp_list: List[float] = []
    seen = set()
    for amp in trigger_amplifications:
        amp_float = float(amp)
        if amp_float not in seen:
            amp_list.append(amp_float)
            seen.add(amp_float)

    poisoned_dirs = {}
    residual_signed_dirs = {}
    residual_abs_dirs = {}

    for amp in amp_list:
        amp_name = amp_to_name(amp)

        poison_dir = os.path.join(poisoned_root_dir, f"amp_{amp_name}")
        res_signed_dir = os.path.join(residual_signed_root_dir, f"amp_{amp_name}")
        res_abs_dir = os.path.join(residual_abs_root_dir, f"amp_{amp_name}")

        os.makedirs(poison_dir, exist_ok=True)
        os.makedirs(res_signed_dir, exist_ok=True)
        os.makedirs(res_abs_dir, exist_ok=True)

        poisoned_dirs[amp] = poison_dir
        residual_signed_dirs[amp] = res_signed_dir
        residual_abs_dirs[amp] = res_abs_dir

    # -------------------------
    # Save selected poisoned training samples.
    # -------------------------
    total = poison_indices.numel()

    if max_save is None or max_save < 0:
        save_count = total
    else:
        save_count = min(max_save, total)

    grid_by_amp: Dict[float, List[torch.Tensor]] = {
        amp: [] for amp in amp_list
    }
    grid_all_amps: List[torch.Tensor] = []

    for order in range(save_count):
        original_idx = int(poison_indices[order].item())

        clean_x, y = train_dataset[original_idx]
        clean_x = tensorize_image(clean_x)
        y_value = int(y.item()) if torch.is_tensor(y) else int(y)

        base_name = f"id_{original_idx:06d}_label_{y_value}_order_{order:06d}"

        save_image(
            clean_x,
            os.path.join(clean_dir, f"{base_name}_clean.png"),
        )

        amplified_images = {}

        for amp in amp_list:
            amp_name = amp_to_name(amp)

            poisoned_amp_x = make_triggered_batch(
                x=clean_x.unsqueeze(0),
                delta_shared=delta_shared,
                amplification=amp,
            ).squeeze(0)
            poisoned_amp_x = poisoned_amp_x.detach().cpu().clamp(0.0, 1.0)
            amplified_images[amp] = poisoned_amp_x

            save_image(
                poisoned_amp_x,
                os.path.join(
                    poisoned_dirs[amp],
                    f"{base_name}_poisoned_amp_{amp_name}.png",
                ),
            )

            save_residual_visualizations(
                clean_x=clean_x,
                poisoned_x=poisoned_amp_x,
                signed_path=os.path.join(
                    residual_signed_dirs[amp],
                    f"{base_name}_residual_signed_amp_{amp_name}.png",
                ),
                abs_path=os.path.join(
                    residual_abs_dirs[amp],
                    f"{base_name}_residual_abs_amp_{amp_name}.png",
                ),
            )

            if order < grid_count:
                grid_by_amp[amp].append(clean_x)
                grid_by_amp[amp].append(poisoned_amp_x)

        if order < grid_count:
            grid_all_amps.append(clean_x)
            for amp in amp_list:
                grid_all_amps.append(amplified_images[amp])

    for amp in amp_list:
        if len(grid_by_amp[amp]) > 0:
            amp_name = amp_to_name(amp)
            grid = torch.stack(grid_by_amp[amp], dim=0)
            save_image(
                grid,
                os.path.join(
                    sample_dir,
                    f"clean_poisoned_comparison_grid_amp_{amp_name}.png",
                ),
                nrow=2,
            )

    if len(grid_all_amps) > 0:
        grid = torch.stack(grid_all_amps, dim=0)
        save_image(
            grid,
            os.path.join(sample_dir, "clean_poisoned_comparison_grid_all_amps.png"),
            nrow=len(amp_list) + 1,
        )

    print(
        f"[*] Saved {save_count} clean training samples, amplified poisoned "
        f"training samples, and residual visualizations for amplification "
        f"factors {amp_list} to: {sample_dir}"
    )

    # -------------------------
    # Save randomly selected triggered test samples by class.
    # -------------------------
    if test_dataset is None or test_labels is None:
        return

    ratio = float(test_save_ratio_per_class)
    if ratio <= 0:
        print("[*] Skipping poisoned test sample saving because test_save_ratio_per_class <= 0.")
        return

    test_root_dir = os.path.join(sample_dir, "test")
    test_clean_root_dir = os.path.join(test_root_dir, "clean")
    test_poisoned_root_dir = os.path.join(test_root_dir, "poisoned")
    test_residual_signed_root_dir = os.path.join(test_root_dir, "residual_signed")
    test_residual_abs_root_dir = os.path.join(test_root_dir, "residual_abs")

    os.makedirs(test_clean_root_dir, exist_ok=True)
    os.makedirs(test_poisoned_root_dir, exist_ok=True)
    os.makedirs(test_residual_signed_root_dir, exist_ok=True)
    os.makedirs(test_residual_abs_root_dir, exist_ok=True)

    labels = test_labels.detach().cpu().long() if torch.is_tensor(test_labels) else torch.tensor(test_labels).long()
    unique_labels = sorted(int(v) for v in labels.unique().tolist())

    generator = torch.Generator()
    generator.manual_seed(int(test_sample_seed))

    selected_test_count = 0

    for class_label in unique_labels:
        class_indices = (labels == class_label).nonzero(as_tuple=False).squeeze(1)
        class_total = int(class_indices.numel())
        if class_total == 0:
            continue

        class_save_count = int(round(class_total * ratio))
        class_save_count = max(1, min(class_save_count, class_total))

        perm = torch.randperm(class_total, generator=generator)
        selected_indices = class_indices[perm[:class_save_count]]

        class_name = label_to_name(class_label)
        clean_class_dir = os.path.join(test_clean_root_dir, class_name)
        os.makedirs(clean_class_dir, exist_ok=True)

        test_poisoned_dirs = {}
        test_residual_signed_dirs = {}
        test_residual_abs_dirs = {}

        for amp in amp_list:
            amp_name = amp_to_name(amp)

            poisoned_class_dir = os.path.join(
                test_poisoned_root_dir,
                f"amp_{amp_name}",
                class_name,
            )
            residual_signed_class_dir = os.path.join(
                test_residual_signed_root_dir,
                f"amp_{amp_name}",
                class_name,
            )
            residual_abs_class_dir = os.path.join(
                test_residual_abs_root_dir,
                f"amp_{amp_name}",
                class_name,
            )

            os.makedirs(poisoned_class_dir, exist_ok=True)
            os.makedirs(residual_signed_class_dir, exist_ok=True)
            os.makedirs(residual_abs_class_dir, exist_ok=True)

            test_poisoned_dirs[amp] = poisoned_class_dir
            test_residual_signed_dirs[amp] = residual_signed_class_dir
            test_residual_abs_dirs[amp] = residual_abs_class_dir

        for order, idx_tensor in enumerate(selected_indices):
            test_idx = int(idx_tensor.item())
            clean_x, y = test_dataset[test_idx]
            clean_x = tensorize_image(clean_x)
            y_value = int(y.item()) if torch.is_tensor(y) else int(y)

            base_name = f"id_{test_idx:06d}_label_{y_value}_order_{order:06d}"

            save_image(
                clean_x,
                os.path.join(clean_class_dir, f"{base_name}_clean.png"),
            )

            for amp in amp_list:
                amp_name = amp_to_name(amp)
                poisoned_amp_x = make_triggered_batch(
                    x=clean_x.unsqueeze(0),
                    delta_shared=delta_shared,
                    amplification=amp,
                ).squeeze(0)
                poisoned_amp_x = poisoned_amp_x.detach().cpu().clamp(0.0, 1.0)

                save_image(
                    poisoned_amp_x,
                    os.path.join(
                        test_poisoned_dirs[amp],
                        f"{base_name}_poisoned_amp_{amp_name}.png",
                    ),
                )

                save_residual_visualizations(
                    clean_x=clean_x,
                    poisoned_x=poisoned_amp_x,
                    signed_path=os.path.join(
                        test_residual_signed_dirs[amp],
                        f"{base_name}_residual_signed_amp_{amp_name}.png",
                    ),
                    abs_path=os.path.join(
                        test_residual_abs_dirs[amp],
                        f"{base_name}_residual_abs_amp_{amp_name}.png",
                    ),
                )

        selected_test_count += int(selected_indices.numel())

    print(
        f"[*] Saved {selected_test_count} triggered test samples "
        f"({ratio * 100:.2f}% per class, rounded with at least one sample/class) "
        f"for amplification factors {amp_list} to: {test_root_dir}"
    )


# -----------------------------
# Victim evaluation + ASR
# -----------------------------

@torch.no_grad()
def evaluate_clean_target_and_asr_multi(
        victim: VictimNet,
        test_dataset: Dataset,
        test_labels: torch.Tensor,
        target_label: int,
        delta_shared: torch.Tensor,
        batch_size: int,
        device: torch.device,
        trigger_amplifications: List[float],
) -> Dict[str, object]:
    victim.eval()

    clean_loss_meter = AverageMeter()
    clean_correct = 0
    clean_total = 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)


        logits = victim(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1)

        clean_loss_meter.update(loss.item(), x.size(0))
        clean_correct += (pred == y).sum().item()
        clean_total += y.numel()

    clean_acc = clean_correct / max(clean_total, 1)

    target_idx = (test_labels == target_label).nonzero(as_tuple=False).squeeze(1)
    target_ds = torch.utils.data.Subset(test_dataset, target_idx.tolist())

    target_loader = DataLoader(
        target_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    tar_correct = 0
    tar_total = 0

    for xb, yb in target_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = victim(xb).argmax(dim=1)

        tar_correct += (pred == yb).sum().item()
        tar_total += yb.numel()

    target_acc = tar_correct / max(tar_total, 1)

    non_idx = (test_labels != target_label).nonzero(as_tuple=False).squeeze(1)
    non_ds = torch.utils.data.Subset(test_dataset, non_idx.tolist())

    asr_dict: Dict[float, float] = {}

    for amp_factor in trigger_amplifications:
        non_loader = DataLoader(
            non_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        asr_success = 0
        asr_total = 0

        for xb, _ in non_loader:
            xb = xb.to(device, non_blocking=True)

            xb_bd = make_triggered_batch(
                x=xb,
                delta_shared=delta_shared,
                amplification=amp_factor,
            )


            pred = victim(xb_bd).argmax(dim=1)

            asr_success += (pred == target_label).sum().item()
            asr_total += xb.size(0)

        asr_dict[float(amp_factor)] = asr_success / max(asr_total, 1)

    return {
        "clean_loss": clean_loss_meter.avg,
        "clean_acc": clean_acc,
        "target_acc": target_acc,
        "asr": asr_dict,
    }


# -----------------------------
# Victim training with ASR logging
# -----------------------------

def train_victim(
        victim: VictimNet,
        train_loader: DataLoader,
        test_loader: DataLoader,
        test_dataset: Dataset,
        test_labels: torch.Tensor,
        target_label: int,
        delta: torch.Tensor,
        trigger_amplifications: List[float],
        eval_every: int,
        device: torch.device,
        epochs: int,
        lr: float,
        weight_decay: float,
        amp: bool,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.SGD(
        victim.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )


    scaler = make_grad_scaler(device, amp)
    amp_enabled = amp and device.type == "cuda"

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "target_acc": [],
    }

    for amp_factor in trigger_amplifications:
        history[f"asr_amp_{amp_factor}"] = []

    for epoch in range(1, epochs + 1):
        victim.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = victim(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()


            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))

        scheduler.step()

        test_loss, test_acc = evaluate_classifier_loss_and_acc(
            victim,
            test_loader,
            device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(loss_meter.avg)
        history["train_acc"].append(acc_meter.avg)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        should_eval_asr = (epoch % eval_every == 0) or (epoch == epochs)

        if should_eval_asr:
            asr_metrics = evaluate_clean_target_and_asr_multi(
                victim=victim,
                test_dataset=test_dataset,
                test_labels=test_labels,
                target_label=target_label,
                delta_shared=delta,
                batch_size=256,
                device=device,
                trigger_amplifications=trigger_amplifications,
            )

            history["target_acc"].append(asr_metrics["target_acc"])

            asr_text = []

            for amp_factor in trigger_amplifications:
                asr_val = asr_metrics["asr"][float(amp_factor)]
                history[f"asr_amp_{amp_factor}"].append(asr_val)
                asr_text.append(f"ASR@x{amp_factor}={asr_val * 100:.2f}%")

            print(
                f"[Victim][Epoch {epoch:03d}/{epochs}] "
                f"train_loss={loss_meter.avg:.4f} "
                f"train_acc={acc_meter.avg * 100:.2f}% "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc * 100:.2f}% "
                f"target_acc={asr_metrics['target_acc'] * 100:.2f}% "
                + " ".join(asr_text)
            )

        else:
            history["target_acc"].append(float("nan"))

            for amp_factor in trigger_amplifications:
                history[f"asr_amp_{amp_factor}"].append(float("nan"))

            print(
                f"[Victim][Epoch {epoch:03d}/{epochs}] "
                f"train_loss={loss_meter.avg:.4f} "
                f"train_acc={acc_meter.avg * 100:.2f}% "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc * 100:.2f}%"
            )

    return history



def save_history_csv(history: Dict[str, List[float]], save_path: str) -> None:
    keys = list(history.keys())
    rows = zip(*[history[k] for k in keys])

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def plot_training_curves(
        history: Dict[str, List[float]],
        trigger_amplifications: List[float],
        save_path: str,
) -> None:
    epochs = history["epoch"]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Victim Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, [v * 100 for v in history["train_acc"]], label="Train Acc")
    plt.plot(epochs, [v * 100 for v in history["test_acc"]], label="Test Acc")
    plt.plot(
        epochs,
        [v * 100 if not math.isnan(v) else float("nan") for v in history["target_acc"]],
        label="Target Acc",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Victim Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)

    for amp_factor in trigger_amplifications:
        key = f"asr_amp_{amp_factor}"
        plt.plot(
            epochs,
            [v * 100 if not math.isnan(v) else float("nan") for v in history[key]],
            label=f"ASR x{amp_factor}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("ASR (%)")
    plt.title("Test ASR with Amplification")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MISA"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./runs/shared_trigger_backdoor")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saved surrogate models",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=False)

    # Attack target / poison settings
    parser.add_argument(
        "--target_label",
        type=int,
        default=0,
        help="Target class. 0=airplane for CIFAR-10.",
    )
    parser.add_argument(
        "--poison_ratio_target",
        type=float,
        default=0.02,
        help=(
            "If < 1, fraction of target-class train samples to poison. "
            "If >= 1, absolute number of target-class samples to poison."
        ),
    )


    parser.add_argument(
        "--trigger_amplification",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help=(
            "List of test-time amplification factors, e.g. "
            "--trigger_amplification 1.0 1.5 2.0"
        ),
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluate ASR every N victim epochs.",
    )

    # Surrogate training
    # parser.add_argument("--sur_epochs", type=int, default=40)
    parser.add_argument("--sur_epochs", type=int, default=200)
    parser.add_argument("--sur_bs", type=int, default=256)
    parser.add_argument("--sur_lr", type=float, default=0.1)
    parser.add_argument("--sur_wd", type=float, default=5e-4)

    # Trigger optimization
    parser.add_argument("--trig_steps", type=int, default=500)
    parser.add_argument("--trig_bs", type=int, default=256)
    parser.add_argument("--trig_lr_delta", type=float, default=0.03)
    parser.add_argument(
        "--eps",
        type=float,
        default=16 / 255.0,
        help="L-infinity bound for the shared trigger.",
    )
    # Victim training
    parser.add_argument("--vic_epochs", type=int, default=80)
    parser.add_argument("--vic_bs", type=int, default=256)
    parser.add_argument("--vic_lr", type=float, default=0.1)
    parser.add_argument("--vic_wd", type=float, default=5e-4)
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
        ],
    )

    # Poisoned image saving
    parser.add_argument(
        "--no_save_poisoned_images",
        action="store_true",
        default=False,
        help="Disable saving clean/poisoned pre-augmentation images.",
    )
    parser.add_argument(
        "--max_save_poisoned",
        type=int,
        default=-1,
        help=(
            "Maximum number of poisoned samples to save. "
            "Use -1 to save all poisoned samples."
        ),
    )
    parser.add_argument(
        "--poisoned_grid_count",
        type=int,
        default=32,
        help="Number of clean/poisoned pairs to include in the comparison grid.",
    )
    parser.add_argument(
        "--test_poisoned_sample_ratio",
        type=float,
        default=0.10,
        help=(
            "Fraction of test samples to save from each class after applying "
            "the trigger. Use 0 to disable saving triggered test samples."
        ),
    )
    parser.add_argument(
        "--test_poisoned_sample_seed",
        type=int,
        default=42,
        help="Random seed used to select test samples for visualization.",
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"AMP enabled: {args.amp}")
    print(f"Shared trigger eps: {args.eps}")
    print(f"Test-time trigger amplification factors: {args.trigger_amplification}")

    num_classes, img_size = get_dataset_info(args.dataset)

    # -----------------------------
    # Data
    # -----------------------------

    test_dataset, test_labels = load_dataset_and_labels(
        root=args.data_root,
        train=False,
        download=True,
        dataset_name=args.dataset,
    )

    surrogate_train_ds = build_surrogate_train_dataset(
        root=args.data_root,
        dataset_name=args.dataset,
        download=True,
    )

    surrogate_train_loader = DataLoader(
        surrogate_train_ds,
        batch_size=args.sur_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    surrogate_test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_dataset, train_labels = load_dataset_and_labels(
        root=args.data_root,
        train=True,
        download=True,
        dataset_name=args.dataset,
    )

    # -----------------------------
    # Train / Load surrogate
    # -----------------------------

    surrogate = StandardSurrogate(
        num_classes=num_classes,
        dataset_name=args.dataset,
    ).to(device)

    surrogate_ckpt_name = (
        f"surrogate_standard_{args.dataset}_resnet18_"
        f"seed{args.seed}_ep{args.sur_epochs}_lr{args.sur_lr}_wd{args.sur_wd}.pt"
    )
    surrogate_ckpt_path = os.path.join(args.ckpt_dir, surrogate_ckpt_name)

    if os.path.exists(surrogate_ckpt_path):
        print(f"[*] Found cached surrogate model at {surrogate_ckpt_path}. Loading...")
        surrogate.load_state_dict(
            safe_torch_load(surrogate_ckpt_path, map_location=device)
        )
    else:
        print("[*] Training surrogate model from scratch...")

        train_surrogate_standard(
            model=surrogate,
            train_loader=surrogate_train_loader,
            test_loader=surrogate_test_loader,
            device=device,
            epochs=args.sur_epochs,
            lr=args.sur_lr,
            weight_decay=args.sur_wd,
            amp=args.amp,
        )

        print(f"[*] Saving trained surrogate to {surrogate_ckpt_path}")
        torch.save(surrogate.state_dict(), surrogate_ckpt_path)

    # -----------------------------
    # Optimize the universal target-margin trigger
    # -----------------------------

    trigger_state = optimize_trigger(
        surrogate=surrogate,
        train_dataset=train_dataset,
        train_labels=train_labels,
        target_label=args.target_label,
        device=device,
        steps=args.trig_steps,
        batch_size=args.trig_bs,
        lr_delta=args.trig_lr_delta,
        eps=args.eps,
        amp=args.amp,
        img_size=img_size,
    )

    np.save(
        os.path.join(args.save_dir, f"optimized_trigger_{args.dataset}_{args.eps}"),
        trigger_state.delta_shared.numpy(),
    )

    # -----------------------------
    # Select poisons and build poisoned training set
    # -----------------------------

    target_idx = (train_labels == args.target_label).nonzero(as_tuple=False).squeeze(1)

    if target_idx.numel() == 0:
        raise ValueError(f"No target-class samples found for target_label={args.target_label}")

    num_target_samples = target_idx.size(0)

    if args.poison_ratio_target < 0:
        raise ValueError("poison_ratio_target must be non-negative.")

    if args.poison_ratio_target >= 1:
        if args.poison_ratio_target > num_target_samples:
            raise ValueError(
                "Number of poisoned samples exceeds number of target-class samples. "
                f"Requested={args.poison_ratio_target}, available={num_target_samples}"
            )
        num_poison = int(args.poison_ratio_target)
    else:
        num_poison = int(round(args.poison_ratio_target * num_target_samples))

    if args.poison_ratio_target < 1:
        print(
            f"Target class size: {num_target_samples}, "
            f"poisoning {num_poison} samples "
            f"({100 * args.poison_ratio_target:.2f}% of target class)"
        )
    else:
        print(
            f"Target class size: {num_target_samples}, "
            f"poisoning {num_poison} samples of target class"
        )

    if num_poison == 0:
        print("[*] poison_ratio_target=0: using clean training set with no poisoned samples.")
        poison_indices = torch.empty(0, dtype=torch.long)
        poisoned_train_ds = train_dataset
    else:
        print("[*] Using RANDOM sample selection for poisoning.")
        perm = torch.randperm(num_target_samples)
        poison_indices = target_idx[perm[:num_poison]]

        poisoned_train_ds = build_poisoned_training_set(
            train_dataset=train_dataset,
            poison_indices=poison_indices,
            delta_shared=trigger_state.delta_shared,
            batch_size=256,
            device=device,
        )

    print(f"First 10 poisoned indices: {poison_indices[:10].tolist()}")

    if num_poison > 0 and not args.no_save_poisoned_images:
        save_poisoned_images(
            train_dataset=train_dataset,
            poison_indices=poison_indices,
            delta_shared=trigger_state.delta_shared,
            trigger_amplifications=args.trigger_amplification,
            save_dir=args.save_dir,
            max_save=args.max_save_poisoned,
            grid_count=args.poisoned_grid_count,
            test_dataset=test_dataset,
            test_labels=test_labels,
            test_save_ratio_per_class=args.test_poisoned_sample_ratio,
            test_sample_seed=args.test_poisoned_sample_seed,
        )

    attack_ckpt = {
        "attack_type": "shared_trigger_clean_label",
        "delta": trigger_state.delta_shared,
        "target_label": args.target_label,
        "eps": trigger_state.eps,
        "poison_indices": poison_indices,
        "trigger_amplification": args.trigger_amplification,
        "dataset": args.dataset,
        "args": vars(args),
    }

    torch.save(
        attack_ckpt,
        os.path.join(args.save_dir, "attack_artifacts.pt"),
    )

    # -----------------------------
    # Train victim
    # -----------------------------

    train_transform = get_train_transform(args.dataset)

    victim_train_ds = DatasetWithTransform(
        poisoned_train_ds,
        transform=train_transform,
    )

    victim_test_ds = DatasetWithTransform(test_dataset)

    victim_train_loader = DataLoader(
        victim_train_ds,
        batch_size=args.vic_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    victim_test_loader = DataLoader(
        victim_test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    victim = VictimNet(
        num_classes=num_classes,
        dataset_name=args.dataset,
        model_name=args.model_name,
    ).to(device)

    history = train_victim(
        victim=victim,
        train_loader=victim_train_loader,
        test_loader=victim_test_loader,
        test_dataset=test_dataset,
        test_labels=test_labels,
        target_label=args.target_label,
        delta=trigger_state.delta_shared,
        trigger_amplifications=args.trigger_amplification,
        eval_every=args.eval_every,
        device=device,
        epochs=args.vic_epochs,
        lr=args.vic_lr,
        weight_decay=args.vic_wd,
        amp=args.amp,
    )

    # -----------------------------
    # Final evaluation
    # -----------------------------

    final_metrics = evaluate_clean_target_and_asr_multi(
        victim=victim,
        test_dataset=test_dataset,
        test_labels=test_labels,
        target_label=args.target_label,
        delta_shared=trigger_state.delta_shared,
        batch_size=256,
        device=device,
        trigger_amplifications=args.trigger_amplification,
    )

    print("\n===== Final Results =====")
    print(f"Clean Loss  : {final_metrics['clean_loss']:.4f}")
    print(f"Clean ACC   : {final_metrics['clean_acc'] * 100:.2f}%")
    print(f"Target ACC  : {final_metrics['target_acc'] * 100:.2f}%")

    for amp_factor in args.trigger_amplification:
        print(
            f"ASR x{amp_factor:<4}: "
            f"{final_metrics['asr'][float(amp_factor)] * 100:.2f}%"
        )

    torch.save(
        victim.state_dict(),
        os.path.join(args.save_dir, "victim_final.pt"),
    )

    torch.save(
        surrogate.state_dict(),
        os.path.join(args.save_dir, "surrogate_standard_final.pt"),
    )

    save_history_csv(
        history,
        os.path.join(args.save_dir, "victim_history.csv"),
    )

    plot_training_curves(
        history,
        args.trigger_amplification,
        os.path.join(args.save_dir, "victim_training_curves.png"),
    )

    with open(os.path.join(args.save_dir, "final_metrics_ce.txt"), "w", encoding="utf-8") as f:
        f.write(f"clean_loss: {final_metrics['clean_loss']}\n")
        f.write(f"clean_acc: {final_metrics['clean_acc']}\n")
        f.write(f"target_acc: {final_metrics['target_acc']}\n")

        for amp_factor in args.trigger_amplification:
            f.write(
                f"asr_amp_{amp_factor}: "
                f"{final_metrics['asr'][float(amp_factor)]}\n"
            )


if __name__ == "__main__":
    main()