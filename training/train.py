"""
@authors: A Bhattacharya
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the training routine that was used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import os, sys
from os.path import join as opj
import numpy as np
import torch
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from dataloading import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import model as model_library

# NOTE this suppresses tensorflow warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getpass
uname = getpass.getuser()

# a class that trains an network to predict actions from depth images
# trainer can be loaded in two ways:
# 1. just for dataloading, in which case dataset_name is provided and usually no_model=True, or
# 2. for model training, in which case just args is provided
class TRAINER:
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            self.device = args.device
            self.basedir = args.basedir
            self.logdir = args.logdir
            self.datadir = args.datadir
            self.ws_suffix = args.ws_suffix
            self.dataset_name = args.dataset
            self.short = args.short

            self.model_type = args.model_type
            self.val_split = args.val_split
            self.seed = args.seed # if args.seed>0 else None
            self.load_checkpoint = args.load_checkpoint
            self.checkpoint_path = args.checkpoint_path
            self.lr = args.lr
            self.N_eps = args.N_eps
            self.lr_warmup_epochs = args.lr_warmup_epochs
            self.lr_decay = args.lr_decay
            self.save_model_freq = args.save_model_freq
            self.val_freq = args.val_freq

        else:
            raise Exception("Args are not provided")


        assert self.dataset_name is not None, 'Dataset name not provided, neither through args nor through dataset_name kwarg'

        ###############
        ## Workspace ##
        ###############

        expname = datetime.now().strftime('d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname + '_' + str(wkspc_ctr))
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        self.writer = SummaryWriter(self.workspace)

        # save ordered args, config, and a logfile to write stdout to
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
                f = opj(self.workspace, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        self.mylogger('[LearnerLSTM init] Making workspace {}'.format(self.workspace))

        self.dataset_dir = opj(self.datadir, self.dataset_name)

        #################
        ## Dataloading ##
        #################

        if self.load_checkpoint:
            print('[LearnerLSTM init] Loading train_val_dirs from checkpoint')
            try:
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
            except:
                print('[LearnerLSTM init] Could not load train_val_dirs from checkpoint, dataloading from scratch')
                train_val_dirs = None
        else:    
            train_val_dirs = None

        self.dataloader(val_split=self.val_split, short=self.short, seed=self.seed, train_val_dirs=train_val_dirs)

        # TODO hardcoding num_training_steps to be the number of trajectories instead of number of images
        self.num_training_steps = self.train_trajlength.shape[0]
        self.num_val_steps = self.val_trajlength.shape[0]
        self.lr_warmup_iters = self.lr_warmup_epochs * self.num_training_steps

        ##################################
        ## Define network and optimizer ##
        ##################################

        self.mylogger('[SETUP] Establishing model and optimizer.')
        if self.model_type == 'LSTMNet':
            self.model = model_library.LSTMNet().to(self.device).float()
        elif self.model_type == 'ConvNet':
            self.model = model_library.ConvNet().to(self.device).float()
        elif self.model_type == 'ViT':            
            self.model = model_library.ViT().to(self.device).float()
        elif self.model_type == 'ViTLSTM':
            self.model = model_library.LSTMNetVIT().to(self.device).float()
        elif self.model_type == 'UNet':
            self.model = model_library.UNetConvLSTMNet().to(self.device).float()
        elif self.model_type == 'DroneMamba':
            # 使用 SSM 进行时序建模 (纯 Mamba 版本)
            self.model = model_library.DroneMamba(use_temporal_ssm=True, d_state=8, hidden_size=128).to(self.device).float()
        elif self.model_type == 'DroneMamba_LSTM':
            # 使用 LSTM 进行时序建模 (混合版本)
            self.model = model_library.DroneMamba(use_temporal_ssm=False, d_state=8, hidden_size=128).to(self.device).float()
        else:
            self.mylogger('[SETUP] Invalid model_type {}. Exiting.'.format(self.model_type))
            exit()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.num_eps_trained = 0
        if self.load_checkpoint:
            self.load_from_checkpoint(self.checkpoint_path)

        self.total_its = self.num_eps_trained * self.num_training_steps

    def mylogger(self, msg):
        print(msg)
        self.logfile.write(msg+'\n')

    def load_from_checkpoint(self, checkpoint_path):
        try:
            self.num_eps_trained = int(checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            self.mylogger('[SETUP] Could not parse number of epochs trained from checkpoint path {}, using 0'.format(checkpoint_path))
        self.mylogger('[SETUP] Loading checkpoint from {}, already trained for {} epochs'.format(checkpoint_path, self.num_eps_trained))
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def dataloader(self, val_split, short=0, seed=None, train_val_dirs=None):
        self.mylogger('[DATALOADER] Loading from {}'.format(self.dataset_dir))
        train_data, val_data, is_png, (self.train_dirs, self.val_dirs) = dataloader(opj(self.basedir, self.dataset_dir), val_split=val_split, short=short, seed=seed, train_val_dirs=train_val_dirs)
        self.train_meta, self.train_ims, self.train_trajlength, self.train_desvel, self.train_currquat, self.train_currctbr = train_data
        self.val_meta, self.val_ims, self.val_trajlength, self.val_desvel, self.val_currquat, self.val_currctbr = val_data
        self.mylogger('[DATALOADER] Dataloading done | train images {}, val images {}'.format(self.train_ims.shape, self.val_ims.shape))

        self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr = preload((self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr), self.device)
        self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr = preload((self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr), self.device)
        self.mylogger('[DATALOADER] Preloading into device {} done'.format(self.device))

        assert self.train_ims.max() <= 1.0 and self.train_ims.min() >= 0.0, 'Images not normalized (values outside [0.0, 1.0])'
        assert self.train_ims.max() > 0.50, "Images not normalized (values only below 0.10, possibly due to not normalizing images from 'old' dataset)"

        self.train_velcmd = self.train_meta[:, range(13, 16) if is_png else range(12, 15)]
        self.val_velcmd = self.val_meta[:, range(13, 16) if is_png else range(12, 15)]

        # save train and val dirs in workspace for later use
        np.save(opj(self.workspace, 'train_val_dirs.npy'), np.array((self.train_dirs, self.val_dirs), dtype=object))

    def lr_scheduler(self, it):
        if it < self.lr_warmup_iters:
            lr = (0.9*self.lr)/self.lr_warmup_iters * it + 0.1*self.lr
        else:
            if self.lr_decay:
                lr = self.lr * (0.1 ** ((it-self.lr_warmup_iters) / (self.N_eps*self.num_training_steps)))
            else:
                lr = self.lr
        return lr

    def save_model(self, ep):
        self.mylogger('[SAVE] Saving model at epoch {}'.format(ep))
        path = self.workspace
        torch.save(self.model.state_dict(), opj(path, 'model_{}.pth'.format(str(ep).zfill(6))))
        self.mylogger('[SAVE] Model saved at {}'.format(path))

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)


    def train(self):

        self.mylogger('[TRAIN] Training for {} epochs'.format(self.N_eps))
        train_start = time.time()

        # starting indices of trajectories in dataset
        self.train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        train_traj_lengths = self.train_trajlength

        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):

            # periodically save model checkpoint
            if ep % self.save_model_freq == 0 and ep - self.num_eps_trained > 0:
                self.save_model(ep)

            # periodically evaluate on validation set
            if ep % self.val_freq == 0:
                self.validation(ep)

            ep_loss = 0
            gradnorm = 0

            # shuffling order of training data trajectories here
            # since we index data using train_traj_starts, we can just shuffle that!
            shuffled_traj_indices = np.random.permutation(len(self.train_traj_starts))
            train_traj_starts = self.train_traj_starts[shuffled_traj_indices]
            train_traj_lengths = self.train_trajlength[shuffled_traj_indices]

            ### Training loop ###
            self.model.train()
            for it in range(self.num_training_steps):
                self.optimizer.zero_grad()
                traj_input = self.train_ims[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it], :, :].unsqueeze(1) #1, traj, 
                desvel = self.train_desvel[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it]].view(-1, 1)
                currquat = self.train_currquat[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it]]
                pred, _ = self.model([traj_input, desvel, currquat]) #, (init_hidden_state, init_cell_state)])
                cmd = self.train_velcmd[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it], :]
                cmd_norm = cmd / desvel # normalize each row by each desvel element
                cmd_norm = cmd_norm
                loss = F.mse_loss(cmd_norm, pred)
                ep_loss += loss
                loss.backward()
                gradnorm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)
                self.optimizer.step()
                new_lr = self.lr_scheduler(self.total_its-self.num_eps_trained*self.num_training_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                self.total_its += 1

            ep_loss /= self.num_training_steps
            gradnorm /= self.num_training_steps

            self.mylogger('[TRAIN] Completed epoch {}/{}, ep_loss = {:.6f}, time = {:.2f}s, time/epoch = {:.2f}s'.format(ep + 1, self.num_eps_trained + self.N_eps, ep_loss, time.time() - train_start, (time.time() - train_start)/(ep + 1 - self.num_eps_trained)))

            self.writer.add_scalar('train/loss', ep_loss, ep)
            self.writer.add_scalar('train/gradnorm', gradnorm, ep)
            self.writer.add_scalar('train/lr', new_lr, self.total_its)
            self.writer.flush()

        self.mylogger('[TRAIN] Training complete, total time = {:.2f}s'.format(time.time() - train_start))
        self.save_model(ep)

    def validation(self, ep):

        self.mylogger('[VAL] Validating for val set of size {} images'.format(self.val_ims.shape[0]))

        val_start = time.time()
        it = 1

        with torch.no_grad():

            ep_loss = 0

            # starting index of trajectories in dataset
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            val_traj_starts = np.hstack((val_traj_starts, -1)) # -1 as the end of 

            ### Validation loop ###
            self.model.eval()

            for it in range(self.num_val_steps):
                # init_hidden_state = torch.rand(self.model.lstm.num_layers, 1, 3).to(self.device).float()
                # init_hidden_state[0] = self.train_velcmd[val_traj_starts[it], :].view(1, 1, -1)
                # init_cell_state = torch.rand(self.model.lstm.num_layers, 1, 3).to(self.device).float()

                traj_input = self.val_ims[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it], :, :].unsqueeze(1)
                desvel = self.val_desvel[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it]].view(-1, 1)
                currquat = self.val_currquat[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it]]
                pred, _ = self.model([traj_input, desvel, currquat]) #, (init_hidden_state, init_cell_state)])
                cmd = self.val_velcmd[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it], :]
                cmd_norm = cmd / desvel # normalize each row by each desvel element
                loss = F.mse_loss(cmd_norm, pred)
                ep_loss += loss

            ep_loss /= (it+1)

            self.mylogger('[VAL] Completed validation, val_loss = {:.6f}, time taken = {:.2f} s'.format(ep_loss, time.time() - val_start))
            self.writer.add_scalar('val/loss', ep_loss, ep)

def argparsing():

    import configargparse
    parser = configargparse.ArgumentParser()

    # general params
    parser.add_argument('--config', is_config_file=True, help='config file relative path')
    parser.add_argument('--basedir', type=str, default='/home/{}/agile_ws/src/agile_flight'.format(uname), help='path to repo')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='path to relative logging directory')
    parser.add_argument('--datadir', type=str, default='/home/{}/agile_ws/src/agile_flight'.format(uname), help='path to relative dataset directory')
    
    # experiment-level and learner params
    parser.add_argument('--ws_suffix', type=str, default='', help='suffix if any to workspace name')
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--dataset', type=str, default='5-2', help='name of dataset')
    parser.add_argument('--short', type=int, default=0, help='if nonzero, how many trajectory folders to load')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=None, help='random seed to use for python random, numpy, and torch -- WARNING, probably not fully implemented')
    parser.add_argument('--device', type=str, default='cuda', help='generic cuda device; specific GPU should be specified in CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='whether to load from a model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='/home/{}/agile_ws/src/agile_flight/learner/logs/d05_10_t03_13/model_000499.pth'.format(uname), help='absolute path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='whether to use lr_decay, hardcoded to exponentially decay to 0.01 * lr by end of training')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')

    args = parser.parse_args()
    print('[CONFIGARGPARSE] Parsing args from config file {}'.format(args.config))

    return args

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args = argparsing()
    print(args)

    learner = TRAINER(args)
    learner.train()
