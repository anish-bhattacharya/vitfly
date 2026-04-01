"""
@authors: A Bhattacharya
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the dataloading routine that was used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import cv2
import glob, os, time
from os.path import join as opj
import numpy as np
import torch
import random
import getpass
uname = getpass.getuser()

def dataloader(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None):
    cropHeight = 60
    cropWidth = 90

    if train_val_dirs is not None:
        traj_folders = train_val_dirs[0] + train_val_dirs[1]
        val_split = len(train_val_dirs[1]) / len(traj_folders)
    else:
        traj_folders = sorted(glob.glob(opj(data_dir, '*')))
        random.seed(seed)
        random.shuffle(traj_folders)

    if short > 0:
        assert short <= len(traj_folders), f"short={short} is greater than the number of folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]
    desired_vels = []
    traj_ims_full = []
    traj_meta_full = []
    curr_quats = []    

    start_dataloading = time.time()

    skippedImages = 0
    skippedFolders = 0
    collisionImages = 0
    collisionFolders = 0

    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}, folder # {i+1}/{len(traj_folders)}, time elapsed {time.time()-start_dataloading:.2f}s')
        im_files = sorted(glob.glob(opj(traj_folder, '*.png')))

        # check for empty folder
        if len(im_files) == 0:
            print(f'[DATALOADER] No images in {os.path.basename(traj_folder)}, skipping')
            continue

        csv_file = 'data.csv'
        # float64 is required to read ros timestamps without rounding
        # NOTE not sure if float64 will break training (torch dtypes)
        traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
        traj_meta[:,-1] = np.int32(np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype="bool")[1:,-1])

        # check for collisions in trajectory
        # if traj_meta[:,-1].sum() > 0:
        #     print(f'[DATALOADER] Collision in {os.path.basename(traj_folder)}, skipping')
        #     collisionFolders += 1
        #     collisionImages += int(len(traj_meta[:,0]))
        #     continue

        # check for nan in metadata
        if np.isnan(traj_meta).any():
            print(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping')
            traj_meta = traj_meta[:,:-1]
            

        # read png files and scale them by 255.0 to recover normalized (0, 1) range
        # for npy files, manually normalize them by a set value (0.09 for "old" dataset)
        traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in im_files], dtype=np.float32) / 255.0

        # check for mismatch in number of images and telemetry entries
        if traj_ims.shape[0] != traj_meta.shape[0]:

            # usually the last image may not have a corresponding line of telemetry, so check specifically for that case
            last_im_timestamp = os.path.basename(im_files[-1])[:-4]
            if float(last_im_timestamp) > traj_meta[-1, 1]:
                traj_ims = traj_ims[:-1]
                print(f'[DATALOADER] Extra image found at end of data, cutting it from {os.path.basename(traj_folder)}')
            if traj_ims.shape[0] != traj_meta.shape[0]:
                print(f'[DATALOADER] Number of images and telemetry still do not match in {os.path.basename(traj_folder)}, skipping')
                skippedFolders += 1
                skippedImages += int(len(traj_meta[:,0]))
                continue
        temp = [cv2.resize(img, (cropWidth, cropHeight)) for img in traj_ims]

        traj_ims = np.array(temp)
        for ii in range(traj_meta.shape[0]):
            desired_vels.append(traj_meta[ii, 2])
            q = traj_meta[ii, 3:7]
            rmat = q 
            curr_quats.append(rmat)
        try:
            traj_ims_full.append(traj_ims)
            traj_meta_full.append(traj_meta)
        except:
            print(f'[DATALOADER] {traj_ims.shape}')
            print(f"[DATALOADER] Suspected empty image, folder {os.path.basename(traj_folder)}")

    print(skippedFolders, skippedImages)
    print(collisionFolders, collisionImages)

    print("[ANALYZER] Analyzing the data....")
    traj_lengths = np.array([traj_ims.shape[0] for traj_ims in traj_ims_full])
    traj_ims_full = np.concatenate(traj_ims_full).reshape(-1, cropHeight, cropWidth)
    traj_meta_full = np.concatenate(traj_meta_full).reshape(-1, traj_meta.shape[-1])
    desired_vels = np.array(desired_vels)
    curr_quats = np.array(curr_quats)


    #Col: mean, std
    #row: ct. brx/y/z
    stats_ctbr = np.zeros((4, 2))
    stats_ctbr[0, :] = np.mean(traj_meta_full[:, 16]), np.std(traj_meta_full[:, 16])
    stats_ctbr[1, :] = np.mean(traj_meta_full[:, 17]), np.std(traj_meta_full[:, 17])
    stats_ctbr[2, :] = np.mean(traj_meta_full[:, 18]), np.std(traj_meta_full[:, 18])
    stats_ctbr[3, :] = np.mean(traj_meta_full[:, 19]), np.std(traj_meta_full[:, 19])

    traj_meta_full[:, 16] = (traj_meta_full[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1])
    traj_meta_full[:, 17] = (traj_meta_full[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1])
    traj_meta_full[:, 18] = (traj_meta_full[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1])
    traj_meta_full[:, 19] = (traj_meta_full[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1])

    
    curr_ctbr = traj_meta_full[:, 16:20]

    #Col: mean, std
    #row: ct. brx/y/z
    stats_ctbr = np.zeros((4, 2))
    stats_ctbr[0, :] = np.mean(traj_meta[:, 16]), np.std(traj_meta[:, 16])
    stats_ctbr[1, :] = np.mean(traj_meta[:, 17]), np.std(traj_meta[:, 17])
    stats_ctbr[2, :] = np.mean(traj_meta[:, 18]), np.std(traj_meta[:, 18])
    stats_ctbr[3, :] = np.mean(traj_meta[:, 19]), np.std(traj_meta[:, 19])

    traj_meta[:, 16] = (traj_meta[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1])
    traj_meta[:, 17] = (traj_meta[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1])
    traj_meta[:, 18] = (traj_meta[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1])
    traj_meta[:, 19] = (traj_meta[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1])

    num_val_trajs = int(val_split * len(traj_lengths))
    if num_val_trajs == 0 and val_split > 0 and len(traj_lengths) >= 2:
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

    return (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train, curr_ctbr_train), (traj_meta_val, traj_ims_val, traj_lengths_val, desired_vels_val, curr_quats_val, curr_ctbr_val), 1, (traj_folders[num_val_trajs:], traj_folders[:num_val_trajs])

def parse_meta_str(meta_str):

    meta = torch.zeros_like(meta_str)

    meta_str

    return meta


def preload(items, device='cpu'):

    return [torch.from_numpy(item).to(device).float() for item in items]