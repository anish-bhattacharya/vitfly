"""
Streaming HDF5-based data loader that eliminates memory accumulation.
Processes trajectories in chunks and stores them on disk.
"""

import cv2
import gc
import glob
import os
import time
from os.path import join as opj
import numpy as np
import h5py
import pickle
import getpass

uname = getpass.getuser()
ERROR_LOG_PATH = '/root/vitfly/training/logs/dataloader_errors.log'

def dataloader_streaming(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None, 
                         chunk_size=50, cache_dir=None):
    """
    Memory-efficient streaming data loader using HDF5.
    
    Args:
        data_dir: Directory containing trajectory folders
        val_split: Validation split ratio
        short: Number of trajectories to process (0 = all)
        seed: Random seed for shuffling
        train_val_dirs: Pre-split train/val directories
        chunk_size: Number of trajectories to process before writing to disk
        cache_dir: Directory to store HDF5 cache file
    
    Returns:
        Same format as original dataloader for compatibility
    """
    cropHeight = 60
    cropWidth = 90
    
    # Generate cache filename
    if cache_dir is None:
        cache_dir = os.path.dirname(data_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    h5_cache_path = opj(cache_dir, f"cached_data_h5_short{short}_split{val_split}_seed{seed}.h5")
    pkl_cache_path = opj(cache_dir, f"cached_data_short{short}_split{val_split}_seed{seed}.pkl")
    
    # Try to load from pickle cache first (for compatibility)
    if os.path.exists(pkl_cache_path):
        print(f'[DATALOADER] Loading cached data from {pkl_cache_path}')
        try:
            with open(pkl_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f'[DATALOADER] Successfully loaded cached data')
            return cached_data
        except Exception as e:
            print(f'[DATALOADER] Failed to load pickle cache: {e}')
    
    # Try to load from HDF5 cache
    if os.path.exists(h5_cache_path):
        print(f'[DATALOADER] Loading from HDF5 cache: {h5_cache_path}')
        try:
            return load_from_h5(h5_cache_path, val_split)
        except Exception as e:
            print(f'[DATALOADER] Failed to load H5 cache: {e}, rebuilding')
            os.remove(h5_cache_path)
    
    # Process data and save to HDF5
    print(f'[DATALOADER] Building HDF5 cache: {h5_cache_path}')
    
    if train_val_dirs is not None:
        traj_folders = train_val_dirs[0] + train_val_dirs[1]
        val_split = len(train_val_dirs[1]) / len(traj_folders)
    else:
        traj_folders = sorted(glob.glob(opj(data_dir, '*')))
        import random
        random.seed(seed)
        random.shuffle(traj_folders)
    
    if short > 0:
        assert short <= len(traj_folders), f"short={short} > num_folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]
    
    start_time = time.time()
    
    # Create HDF5 file with chunked storage
    with h5py.File(h5_cache_path, 'w') as h5f:
        # Estimate total size (will resize as needed)
        estimated_samples = len(traj_folders) * 400  # ~400 images per trajectory
        
        # Create resizable datasets
        h5f.create_dataset('images', shape=(0, cropHeight, cropWidth), 
                          maxshape=(None, cropHeight, cropWidth),
                          dtype=np.float32, chunks=(100, cropHeight, cropWidth),
                          compression='gzip', compression_opts=1)
        
        h5f.create_dataset('metadata', shape=(0, 20),  # Assuming 20 metadata columns
                          maxshape=(None, 20),
                          dtype=np.float64, chunks=(1000, 20),
                          compression='gzip', compression_opts=1)
        
        h5f.create_dataset('desired_vels', shape=(0,),
                          maxshape=(None,),
                          dtype=np.float64, chunks=(1000,))
        
        h5f.create_dataset('curr_quats', shape=(0, 4),
                          maxshape=(None, 4),
                          dtype=np.float64, chunks=(1000, 4))
        
        # Track trajectory boundaries for splitting
        traj_lengths = []
        total_samples = 0
        
        skipped_folders = 0
        skipped_images = 0
        
        for folder_idx, traj_folder in enumerate(traj_folders):
            print(f'[DATALOADER] Processing {folder_idx+1}/{len(traj_folders)}: {os.path.basename(traj_folder)}', flush=True)
            
            try:
                # Load images
                im_files = sorted(glob.glob(opj(traj_folder, '*.png')))
                if len(im_files) == 0:
                    print(f'[DATALOADER] No images in {os.path.basename(traj_folder)}, skipping')
                    continue
                
                # Load metadata
                csv_path = opj(traj_folder, 'data.csv')
                traj_meta = np.genfromtxt(csv_path, delimiter=',', dtype=np.float64)[1:]
                traj_meta[:,-1] = np.int32(np.genfromtxt(csv_path, delimiter=',', dtype="bool")[1:,-1])
                
                if np.isnan(traj_meta).any():
                    print(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping')
                    continue
                
                # Process images one at a time
                n_images = len(im_files)
                traj_ims = np.empty((n_images, cropHeight, cropWidth), dtype=np.float32)
                
                for idx, im_file in enumerate(im_files):
                    img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f'Failed to read: {im_file}')
                    resized = cv2.resize(img, (cropWidth, cropHeight))
                    traj_ims[idx] = resized.astype(np.float32) / 255.0
                    del img, resized
                
                # Handle image/metadata mismatch
                if traj_ims.shape[0] != traj_meta.shape[0]:
                    last_im_timestamp = os.path.basename(im_files[-1])[:-4]
                    if float(last_im_timestamp) > traj_meta[-1, 1]:
                        traj_ims = traj_ims[:-1]
                    if traj_ims.shape[0] != traj_meta.shape[0]:
                        print(f'[DATALOADER] Image/metadata mismatch in {os.path.basename(traj_folder)}, skipping')
                        skipped_folders += 1
                        skipped_images += len(traj_meta)
                        continue
                
                # Extract metadata
                desired_vels_traj = traj_meta[:, 2]
                curr_quats_traj = traj_meta[:, 3:7]
                
                # Append to HDF5 (resize datasets)
                current_size = h5f['images'].shape[0]
                new_size = current_size + traj_ims.shape[0]
                
                h5f['images'].resize((new_size, cropHeight, cropWidth))
                h5f['metadata'].resize((new_size, traj_meta.shape[1]))
                h5f['desired_vels'].resize((new_size,))
                h5f['curr_quats'].resize((new_size, 4))
                
                h5f['images'][current_size:new_size] = traj_ims
                h5f['metadata'][current_size:new_size] = traj_meta
                h5f['desired_vels'][current_size:new_size] = desired_vels_traj
                h5f['curr_quats'][current_size:new_size] = curr_quats_traj
                
                traj_lengths.append(traj_ims.shape[0])
                total_samples += traj_ims.shape[0]
                
                # Explicit cleanup
                del traj_ims, traj_meta, desired_vels_traj, curr_quats_traj
                
                # Periodic garbage collection
                if (folder_idx + 1) % chunk_size == 0:
                    gc.collect()
                    elapsed = time.time() - start_time
                    print(f'[DATALOADER] Processed {folder_idx+1}/{len(traj_folders)} trajectories, '
                          f'{total_samples} samples, {elapsed:.1f}s elapsed', flush=True)
                
            except Exception as e:
                error_msg = f'[DATALOADER ERROR] Failed to process {traj_folder}: {type(e).__name__}: {e}'
                print(error_msg, flush=True)
                with open(ERROR_LOG_PATH, 'a') as f:
                    f.write(f'{error_msg}\n')
                continue
        
        # Store trajectory lengths
        h5f.create_dataset('traj_lengths', data=np.array(traj_lengths, dtype=np.int32))
        h5f.attrs['num_trajectories'] = len(traj_lengths)
        h5f.attrs['total_samples'] = total_samples
        h5f.attrs['val_split'] = val_split
        h5f.attrs['skipped_folders'] = skipped_folders
        h5f.attrs['skipped_images'] = skipped_images
        
        print(f'[DATALOADER] HDF5 cache created: {total_samples} samples from {len(traj_lengths)} trajectories')
        print(f'[DATALOADER] Skipped: {skipped_folders} folders, {skipped_images} images')
    
    # Load from HDF5 and return in original format
    return load_from_h5(h5_cache_path, val_split)


def load_from_h5(h5_path, val_split):
    """Load data from HDF5 and return in original dataloader format."""
    print(f'[DATALOADER] Loading from HDF5: {h5_path}')
    
    with h5py.File(h5_path, 'r') as h5f:
        # Load all data into memory (HDF5 provides efficient chunked loading)
        traj_ims_full = h5f['images'][:]
        traj_meta_full = h5f['metadata'][:]
        desired_vels = h5f['desired_vels'][:]
        curr_quats = h5f['curr_quats'][:]
        traj_lengths = h5f['traj_lengths'][:]
        
        cropHeight, cropWidth = traj_ims_full.shape[1], traj_ims_full.shape[2]
        
        print(f'[DATALOADER] Loaded {traj_ims_full.shape[0]} samples from {len(traj_lengths)} trajectories')
    
    # Normalize ctbr columns (16:20)
    stats_ctbr = np.zeros((4, 2))
    for i in range(4):
        stats_ctbr[i, 0] = np.mean(traj_meta_full[:, 16+i])
        stats_ctbr[i, 1] = np.std(traj_meta_full[:, 16+i])
        traj_meta_full[:, 16+i] = (traj_meta_full[:, 16+i] - stats_ctbr[i, 0]) / (2 * stats_ctbr[i, 1])
    
    curr_ctbr = traj_meta_full[:, 16:20]
    
    # Split into train/val
    num_val_trajs = int(val_split * len(traj_lengths))
    
    if num_val_trajs == 0 and val_split > 0 and len(traj_lengths) >= 2:
        # Sample-level split
        num_val_samples = int(val_split * len(traj_meta_full))
        indices = np.random.permutation(len(traj_meta_full))
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]
        
        traj_meta_train = traj_meta_full[train_indices]
        traj_meta_val = traj_meta_full[val_indices]
        traj_ims_train = traj_ims_full[train_indices]
        traj_ims_val = traj_ims_full[val_indices]
        traj_lengths_train = np.array([len(train_indices)])
        traj_lengths_val = np.array([len(val_indices)])
        desired_vels_train = desired_vels[train_indices]
        desired_vels_val = desired_vels[val_indices]
        curr_quats_train = curr_quats[train_indices]
        curr_quats_val = curr_quats[val_indices]
        curr_ctbr_train = curr_ctbr[train_indices]
        curr_ctbr_val = curr_ctbr[val_indices]
    else:
        # Trajectory-level split
        val_idx = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32) if num_val_trajs > 0 else 0
        
        traj_meta_val = traj_meta_full[:val_idx] if val_idx > 0 else traj_meta_full[:0]
        traj_meta_train = traj_meta_full[val_idx:] if val_idx > 0 else traj_meta_full
        traj_ims_val = traj_ims_full[:val_idx] if val_idx > 0 else traj_ims_full[:0]
        traj_ims_train = traj_ims_full[val_idx:] if val_idx > 0 else traj_ims_full
        traj_lengths_val = traj_lengths[:num_val_trajs] if num_val_trajs > 0 else np.array([])
        traj_lengths_train = traj_lengths[num_val_trajs:] if num_val_trajs > 0 else traj_lengths
        desired_vels_val = desired_vels[:val_idx] if val_idx > 0 else desired_vels[:0]
        desired_vels_train = desired_vels[val_idx:] if val_idx > 0 else desired_vels
        curr_quats_val = curr_quats[:val_idx] if val_idx > 0 else curr_quats[:0]
        curr_quats_train = curr_quats[val_idx:] if val_idx > 0 else curr_quats
        curr_ctbr_val = curr_ctbr[:val_idx] if val_idx > 0 else curr_ctbr[:0]
        curr_ctbr_train = curr_ctbr[val_idx:] if val_idx > 0 else curr_ctbr
    
    print(f'[DATALOADER] Train: {len(traj_ims_train)} samples, Val: {len(traj_ims_val)} samples')
    
    result = (
        (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train, curr_ctbr_train),
        (traj_meta_val, traj_ims_val, traj_lengths_val, desired_vels_val, curr_quats_val, curr_ctbr_val),
        1,
        ([], [])  # Placeholder for folder names
    )
    
    return result
