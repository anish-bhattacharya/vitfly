"""
Lazy-loading PyTorch Dataset for Flightmare data.

This module provides memory-efficient data loading by:
1. Building a file path index without loading images
2. Loading images on-demand in __getitem__
3. Supporting PyTorch DataLoader with parallel workers

Solves the memory exhaustion issue where loading all 580 trajectories
(110K images, 3.4GB) into RAM causes the process to hang.
"""

import cv2
import glob
import os
from os.path import join as opj
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import time


class LazyFlightmareDataset(Dataset):
    """
    Lazy-loading dataset that stores only file paths and loads data on-demand.
    
    Each sample consists of:
    - depth: Grayscale depth image (60x90)
    - velocity: Current 3D velocity from metadata columns 10:13
    - quat: Quaternion from metadata columns 3:7
    - target: Expert velocity command from metadata columns 13:16 (normalized)
    """
    
    def __init__(self, sample_paths, cropHeight=60, cropWidth=90):
        """
        Args:
            sample_paths: List of (image_path, csv_path, row_idx) tuples
            cropHeight: Target height for depth images
            cropWidth: Target width for depth images
        """
        self.sample_paths = sample_paths
        self.cropHeight = cropHeight
        self.cropWidth = cropWidth
        
        # Cache for CSV files to avoid repeated disk reads
        self._csv_cache = {}
        
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        img_path, csv_path, row_idx = self.sample_paths[idx]
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            img = img.astype(np.float32) / 255.0
            img = cv2.resize(img, (self.cropWidth, self.cropHeight))
            depth = torch.from_numpy(img).unsqueeze(0).float()
            
            if csv_path not in self._csv_cache:
                meta = np.genfromtxt(csv_path, delimiter=',', dtype=np.float64)[1:]
                if np.isnan(meta[:, :-1]).any():
                    raise ValueError(f"NaN values in non-collision columns of CSV: {csv_path}")
                self._csv_cache[csv_path] = meta
            else:
                meta = self._csv_cache[csv_path]
            
            if row_idx >= len(meta):
                raise ValueError(f"Row index {row_idx} out of bounds for {csv_path}")
            
            row = meta[row_idx]
            
            velocity = torch.from_numpy(row[10:13]).float()
            quat = torch.from_numpy(row[3:7]).float()
            target = torch.from_numpy(row[13:16]).float()
            
            if torch.isnan(velocity).any() or torch.isnan(quat).any() or torch.isnan(target).any():
                raise ValueError(f"NaN in extracted data at row {row_idx}")
            
            target = target / (torch.norm(target) + 1e-6)
            
            return depth, velocity, quat, target
            
        except Exception as e:
            print(f"[LAZY DATASET] Error loading sample {idx} from {img_path}: {e}")
            fallback_idx = torch.randint(0, len(self.samples), (1,)).item()
            return self.__getitem__(fallback_idx)


def create_lazy_dataloader(data_dir, val_split=0.2, batch_size=32, num_workers=4,
                           short=0, seed=42, pin_memory=True):
    """
    Create lazy-loading DataLoaders for training and validation.
    
    Args:
        data_dir: Path to directory containing trajectory folders
        val_split: Fraction of data to use for validation (0.0-1.0)
        batch_size: Batch size for DataLoader
        num_workers: Number of parallel workers for data loading
        short: If > 0, only use first N trajectories (for testing)
        seed: Random seed for reproducibility
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        (train_loader, val_loader, stats_dict)
    """
    print("[LAZY DATALOADER] Building file path index...")
    start_time = time.time()
    
    # Get all trajectory folders
    traj_folders = sorted(glob.glob(opj(data_dir, '*')))
    
    # Shuffle for random train/val split
    random.seed(seed)
    random.shuffle(traj_folders)
    
    # Limit to subset if requested
    if short > 0:
        assert short <= len(traj_folders), \
            f"short={short} is greater than the number of folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]
    
    print(f"[LAZY DATALOADER] Found {len(traj_folders)} trajectory folders")
    
    # Build sample paths (image_path, csv_path, row_idx)
    all_sample_paths = []
    skipped_folders = 0
    skipped_images = 0
    
    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders) // 10 > 0 and i % (len(traj_folders) // 10) == 0:
            print(f'[LAZY DATALOADER] Indexing folder {os.path.basename(traj_folder)}, '
                  f'folder # {i+1}/{len(traj_folders)}, '
                  f'time elapsed {time.time()-start_time:.2f}s')
        
        # Get image files
        im_files = sorted(glob.glob(opj(traj_folder, '*.png')))
        
        # Check for empty folder
        if len(im_files) == 0:
            print(f'[LAZY DATALOADER] No images in {os.path.basename(traj_folder)}, skipping')
            skipped_folders += 1
            continue
        
        # Check for CSV file
        csv_file = opj(traj_folder, 'data.csv')
        if not os.path.exists(csv_file):
            print(f'[LAZY DATALOADER] No data.csv in {os.path.basename(traj_folder)}, skipping')
            skipped_folders += 1
            continue
        
        # Verify CSV row count matches image count before adding samples
        try:
            meta = np.genfromtxt(csv_file, delimiter=',', dtype=np.float64)[1:]
            if len(im_files) != len(meta):
                print(f'[LAZY DATALOADER] Row count mismatch in {os.path.basename(traj_folder)}: '
                      f'{len(im_files)} images vs {len(meta)} CSV rows, skipping')
                skipped_folders += 1
                skipped_images += len(im_files)
                continue
            
            # Add samples only once
            for row_idx, im_file in enumerate(im_files):
                all_sample_paths.append((im_file, csv_file, row_idx))
                
        except Exception as e:
            print(f'[LAZY DATALOADER] Error processing {os.path.basename(traj_folder)}: {e}')
            skipped_folders += 1
            skipped_images += len(im_files)
            continue
    
    print(f"[LAZY DATALOADER] Indexed {len(all_sample_paths)} samples")
    print(f"[LAZY DATALOADER] Skipped {skipped_folders} folders, {skipped_images} images")
    print(f"[LAZY DATALOADER] Indexing took {time.time()-start_time:.2f}s")
    
    if len(all_sample_paths) == 0:
        raise ValueError(
            f"No valid samples found in {data_dir}. "
            f"Skipped {skipped_folders} folders with {skipped_images} images. "
            "Please check your dataset for valid trajectory folders with PNG images and data.csv files."
        )
    
    num_val_samples = int(val_split * len(all_sample_paths))
    val_sample_paths = all_sample_paths[:num_val_samples]
    train_sample_paths = all_sample_paths[num_val_samples:]
    
    print(f"[LAZY DATALOADER] Train samples: {len(train_sample_paths)}")
    print(f"[LAZY DATALOADER] Val samples: {len(val_sample_paths)}")
    
    train_dataset = LazyFlightmareDataset(train_sample_paths)
    val_dataset = LazyFlightmareDataset(val_sample_paths) if num_val_samples > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    # Compute statistics (for compatibility with old code)
    stats_dict = {
        'num_train_samples': len(train_sample_paths),
        'num_val_samples': len(val_sample_paths),
        'num_trajectories': len(traj_folders) - skipped_folders,
        'skipped_folders': skipped_folders,
        'skipped_images': skipped_images
    }
    
    return train_loader, val_loader, stats_dict
