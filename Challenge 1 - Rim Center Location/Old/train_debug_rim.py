#!/usr/bin/env python3
"""
train_debug_rim.py

Single-file utility to:
 - load labels.json (normalized centers) and images (handles nested folders)
 - Robust dataset with debug logging to a file (debug_log_path)
 - Run an automatic debug check (num_workers=0) that loads a small batch and a few direct samples.
   If debug fails, the script exits with a clear error and log location.
 - Optionally run a short training loop (--train) if debug passes.

Usage examples:
  # Just run debug (default)
  python train_debug_rim.py --images_dir "/path/to/labeled_example_data" --labels "/path/to/labels.json"

  # Run debug then a short training run (useful after debug success)
  python train_debug_rim.py --images_dir "/path/to/labeled_example_data" --labels "/path/to/labels.json" --train --epochs 10 --num_workers 4

Important:
 - The script intentionally runs the debug batch with num_workers=0 to surface errors from __getitem__.
 - If you enable training and want DataLoader workers, the dataset writes per-worker logs to debug_log_path so a crashing worker can be diagnosed.
"""

import argparse
import json
import math
import os
import sys
import traceback
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# Optional: use albumentations if available but keep transforms optional
try:
    import albumentations as A
    HAVE_A = True
except Exception:
    HAVE_A = False

# ------------- Dataset -------------


class RobustWheelHeatmapDataset(Dataset):
    """
    Reads labels.json (list of {filename, is_present, center: [x_norm,y_norm]})
    and builds self.samples with existing files only (searches nested folders).
    If debug_log_path is specified, __getitem__ writes simple start/ok/exception notes there
    so a crashing worker can be diagnosed after the fact.
    """

    def __init__(
        self,
        labels_path: str,
        images_dir: str,
        img_size: int = 128,
        sigma_ratio: float = 0.02,
        augment: bool = False,
        debug_log_path: str = "/tmp/ds_worker_log.txt",
        verbose: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.img_size = int(img_size)
        self.sigma = max(1.0, sigma_ratio * self.img_size)
        self.augment = bool(augment)
        self.debug_log_path = debug_log_path
        self.verbose = bool(verbose)

        with open(labels_path, "r") as f:
            entries = json.load(f)

        self.samples = []
        missing = 0
        for entry in entries:
            fname = entry.get("filename")
            if not fname:
                continue
            candidate = self.images_dir / fname
            if not candidate.exists():
                found = list(self.images_dir.rglob(fname))
                candidate = Path(found[0]) if found else candidate
            if not candidate.exists():
                missing += 1
                if self.verbose and missing <= 20:
                    print(f"[Dataset] missing file: {self.images_dir / fname}")
                continue
            is_present = bool(float(entry.get("is_present", 0)))
            center = entry.get("center", None)
            if is_present and center and len(center) >= 2:
                x_norm, y_norm = float(center[0]), float(center[1])
            else:
                x_norm, y_norm = -1.0, -1.0
            self.samples.append(
                {"imgp": candidate, "x_norm": x_norm, "y_norm": y_norm, "has_wheel": is_present}
            )

        if self.verbose:
            print(
                f"[Dataset] labels_total={len(entries)}; usable_samples={len(self.samples)}; missing={missing}"
            )

        # transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # optional augmentation using albumentations if available
        if self.augment and HAVE_A:
            self.aug_pipeline = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.MotionBlur(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                ]
            )
        else:
            self.aug_pipeline = None
            if self.augment and not HAVE_A and self.verbose:
                print("[Dataset] albumentations not installed; augment disabled.")

    def __len__(self) -> int:
        return len(self.samples)

    def _draw_gaussian(self, heatmap: np.ndarray, center: tuple, sigma: float) -> np.ndarray:
        x, y = center
        h, w = heatmap.shape
        xs = np.arange(0, w, 1, np.float32)
        ys = np.arange(0, h, 1, np.float32)[:, np.newaxis]
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
        return np.maximum(heatmap, g)

    def _log(self, line: str):
        if not self.debug_log_path:
            return
        try:
            with open(self.debug_log_path, "a") as L:
                L.write(line + "\n")
        except Exception:
            # avoid crashing because of logging I/O
            pass

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        imgp = sample["imgp"]
        ts = time.time()
        self._log(f"{ts:.3f}\tSTART\tidx={idx}\tfile={imgp}")
        try:
            # Load image
            img = np.array(Image.open(imgp).convert("RGB"))
            h0, w0 = img.shape[:2]

            # Augmentation if available
            if self.augment and self.aug_pipeline is not None:
                try:
                    img = self.aug_pipeline(image=img)["image"]
                except Exception as e:
                    self._log(f"{time.time():.3f}\tAUG_ERR\tidx={idx}\tfile={imgp}\t{repr(e)}")
                    raise

            # Resize and to tensor
            pil = Image.fromarray(img)
            pil_resized = pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            img_t = self.to_tensor(pil_resized)
            img_t = self.normalize(img_t)

            # make target heatmap
            if sample["has_wheel"] and sample["x_norm"] >= 0:
                cx = float(sample["x_norm"]) * self.img_size
                cy = float(sample["y_norm"]) * self.img_size
                heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                heatmap = self._draw_gaussian(heatmap, (cx, cy), self.sigma)
                presence = 1.0
            else:
                heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                presence = 0.0

            heat_t = torch.from_numpy(heatmap).unsqueeze(0).float()
            presence_t = torch.tensor(presence, dtype=torch.float32)

            self._log(f"{time.time():.3f}\tOK\tidx={idx}\tfile={imgp}")
            return img_t, heat_t, presence_t, str(imgp)

        except Exception as exc:
            self._log(f"{time.time():.3f}\tEXC\tidx={idx}\tfile={imgp}\n{traceback.format_exc()}")
            raise


# ------------- Simple model (lightweight) -------------


class SimpleUNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Very small encoder-decoder for quick tests
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 32, 2, stride=2), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU())

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)  # /2
        e2 = self.enc2(e1)  # /4
        e3 = self.enc3(e2)  # /8
        d3 = self.dec3(e3)  # /4
        d3c = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3c)  # /2
        d2c = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2c)  # /1
        out = self.final(torch.cat([d1, x], dim=1))
        return out


# ------------- Utilities & training -------------


def debug_check_dataset(
    ds: RobustWheelHeatmapDataset, batch_size: int = 8, debug_samples: int = 8, device: str = "cpu"
):
    """
    Run a small direct-sample test and a DataLoader batch load with num_workers=0.
    Raises RuntimeError with helpful message if something fails.
    """
    print(">>> Running dataset direct-sample checks (first few samples)...")
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset contains zero usable samples. Check labels/images paths.")

    # direct iteration (no DataLoader) - shows file-specific exceptions
    for i in range(min(debug_samples, n)):
        try:
            img_t, heat_t, pres_t, path = ds[i]
            print(f" sample[{i}] OK: path={path} shapes: img={tuple(img_t.shape)}, heat={tuple(heat_t.shape)} pres={float(pres_t)}")
        except Exception as e:
            raise RuntimeError(f"Dataset direct sample load failed at index {i}. See dataset debug log: {ds.debug_log_path}\n{traceback.format_exc()}") from e

    # DataLoader batch (num_workers=0) to show full traceback inline
    print(">>> Running DataLoader test (num_workers=0)...")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    try:
        batch = next(iter(dl))
        # batch is (imgs, heats, presences, paths)
        print(" DataLoader batch OK:")
        print("  imgs:", getattr(batch[0], "shape", None))
        print("  heatmaps:", getattr(batch[1], "shape", None))
        print("  presences:", getattr(batch[2], "shape", None))
    except Exception as e:
        raise RuntimeError(
            f"DataLoader batch load failed with num_workers=0. This surfaces the real exception.\n"
            f"Check dataset debug log at: {ds.debug_log_path}\n\n{traceback.format_exc()}"
        ) from e

    print(">>> Debug checks passed successfully.")


def train_short(
    ds,
    out_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    num_workers: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
):
    device = torch.device(device)
    # split
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    model = SimpleUNetSmall(in_ch=3, out_ch=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, heats, pres, paths in train_loader:
            imgs = imgs.to(device)
            heats = heats.to(device)
            preds = model(imgs)
            loss = criterion(preds, heats)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * imgs.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heats, pres, paths in val_loader:
                imgs = imgs.to(device); heats = heats.to(device)
                preds = model(imgs)
                loss = criterion(preds, heats)
                val_loss += float(loss.item()) * imgs.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        print(f"Epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir / "best_model.pth"
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            print(" Saved best model ->", ckpt)


# ------------- CLI & main -------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Directory containing images (can be nested).")
    p.add_argument("--labels", required=True, help="Path to labels.json (normalized centers).")
    p.add_argument("--out_dir", default="out", help="Output folder for artifacts.")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--train", action="store_true", help="If set, run a short training run after debug check.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2, help="Workers for DataLoader during training (debug runs use 0).")
    p.add_argument("--debug_log_path", default="/tmp/ds_worker_log.txt", help="Path where dataset worker logs are appended.")
    p.add_argument("--augment", action="store_true", help="Enable augmentation (requires albumentations).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print("Args:", args)

    # remove existing debug log file to start fresh
    try:
        if args.debug_log_path:
            open(args.debug_log_path, "w").close()
    except Exception:
        pass

    ds = RobustWheelHeatmapDataset(
        labels_path=args.labels,
        images_dir=args.images_dir,
        img_size=args.img_size,
        sigma_ratio=0.02,
        augment=args.augment,
        debug_log_path=args.debug_log_path,
        verbose=True,
    )

    # Run debug checks (will raise RuntimeError with helpful message if anything fails)
    try:
        debug_check_dataset(ds, batch_size=args.batch_size, debug_samples=8, device=args.device)
    except Exception as e:
        print("\n========== DEBUG CHECK FAILED ==========")
        print(str(e))
        print("Dataset debug log (last 200 lines) is at:", args.debug_log_path)
        try:
            with open(args.debug_log_path, "r") as L:
                lines = L.readlines()
                tail = lines[-200:] if len(lines) > 200 else lines
                print("---- debug log tail ----")
                for l in tail:
                    print(l.rstrip())
                print("---- end log tail ----")
        except Exception:
            print("Could not read debug log.")
        sys.exit(2)

    print("\nDEBUG CHECK PASSED. Dataset loads correctly in single-process mode.")

    if args.train:
        print("Starting short training run (you can disable with --train).")
        train_short(
            ds,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=1e-3,
            device=args.device,
        )
        print("Training finished.")
    else:
        print("Exiting after debug check (use --train to run a short training).")


if __name__ == "__main__":
    main()
