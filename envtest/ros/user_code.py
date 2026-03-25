#!/usr/bin/python3

from utils import AgileCommandMode, AgileCommand
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

import glob, os, sys, time
from os.path import join as opj
import rospy

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import *

# 3D line determined by two points (x1, y1, z1) and (x2, y2, z2)
# sphere determined by a center point (x3, y3, z3) and radius r
# quantity b^2 - 4ac < 0 then there is no intersection, where:
# b = 2*( (x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3) )
# a = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
# c = x3^2 + y3^2 + z3^2 + x1^2 + y1^2 + z1^2 - 2*(x3*x1 + y3*y1 + z3*z1) - r^2
# line is a 2-tuple of 3-tuples, obstacle is a 2-tuple of the center 3-tuple and the radius float
def check_collision(line, obstacle):
    (x1, y1, z1), (x2, y2, z2) = line
    (x3, y3, z3), r = obstacle
    b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    c = (
        x3**2
        + y3**2
        + z3**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (x3 * x1 + y3 * y1 + z3 * z1)
        - r**2
    )
    return b**2 - 4 * a * c >= 0


def compute_command_vision_based(state, orig_img, prev_img, desiredVel, trained_model, hidden_state):
    # print("Computing command vision-based!")

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    # command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0
    command.mode = 2
    
    ###############
    ## Load data ##
    ###############

    q = np.array([state.att[0], state.att[1], state.att[2], state.att[3]])
    
    h, w = (60, 90)
    img = cv2.resize(orig_img, (w, h))
    img2 = orig_img.copy() # used for generating debugimg
    img = ToTensor()(np.array(img))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'LSTMNet' in trained_model.__class__.__name__:
        if trained_model.__class__.__name__ == 'LSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 395
        elif trained_model.__class__.__name__ == 'LSTMNetVIT':
            trained_model.lstm.num_layers = 3
            trained_model.lstm.hidden_size = 128
        elif trained_model.__class__.__name__ == 'UNetConvLSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 200
        
        # LSTM 系列模型需要隐藏状态
        if state.pos[0] < 0.5 or hidden_state is None:
            hidden_state = (torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(device),
                           torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(device))
        
        with torch.no_grad():
            output = trained_model([img.view(1, 1, h, w).to(device), 
                                   torch.tensor(desiredVel).view(1, 1).float().to(device).expand(1, 3), 
                                   torch.tensor(q).view(1, -1).float().to(device), 
                                   hidden_state])
            if isinstance(output, tuple):
                x, hidden_state = output
            else:
                x = output
                
    elif 'VMambaLSTM' in trained_model.__class__.__name__:
        # VMamba+LSTM 模型
        if trained_model.lstm.num_layers == 2 and trained_model.lstm.hidden_size == 128:
            pass  # 默认配置
        
        if state.pos[0] < 0.5 or hidden_state is None:
            hidden_state = (torch.zeros(2, 128).float().to(device),
                           torch.zeros(2, 128).float().to(device))
        
        with torch.no_grad():
            output = trained_model([img.view(1, 1, h, w).to(device), 
                                   torch.tensor(desiredVel).view(1, 1).float().to(device).expand(1, 3), 
                                   torch.tensor(q).view(1, -1).float().to(device)],
                                   hidden_state)
            if isinstance(output, tuple):
                x, hidden_state = output
            else:
                x = output
                
    elif 'DroneMamba' in trained_model.__class__.__name__:
        # DroneMamba 模型，使用 SSM 版本不需要隐藏状态
        with torch.no_grad():
            output = trained_model([img.view(1, 1, h, w).to(device), 
                                   torch.tensor(desiredVel).view(1, 1).float().to(device).expand(1, 3), 
                                   torch.tensor(q).view(1, -1).float().to(device)])
            if isinstance(output, tuple):
                x, hidden_state = output
            else:
                x = output
                hidden_state = None
                
    else:
        # 其他模型（包括 ViT, ConvNet 等）
        with torch.no_grad():
            output = trained_model(img.view(1, 1, h, w).to(device))
            if isinstance(output, tuple):
                x, hidden_state = output
            else:
                x = output
                hidden_state = None


    x = x.squeeze().cpu().detach().numpy()
    
    # 调试输出
    if rospy.get_time() - int(rospy.get_time()) < 0.1:
        print(f'[USER_CODE] 模型原始输出 x = {x}, desiredVel = {desiredVel}')
    
    x[0] = np.clip(x[0], -1, 1)
    
    # 确保有足够的前向速度
    # 模型应该输出 [前向，侧向，垂直] 方向的速度分量
    # 归一化后乘以期望速度
    norm_x = np.linalg.norm(x)
    if norm_x > 0.01:
        x = x / norm_x * desiredVel
    else:
        # 如果模型输出接近零，默认给一个小的前向速度
        x = np.array([desiredVel, 0.0, 0.0])
    
    # 确保最小前向速度 - 这是关键修复
    # 即使模型预测的前向速度很小，也要保证至少 2m/s 的前向速度
    min_forward_vel = 2.0
    if x[0] < min_forward_vel:
        # 重新归一化，保持方向但增加前向分量
        forward_ratio = x[0] / (np.linalg.norm(x[1:]) + 1e-6) if np.linalg.norm(x[1:]) > 0 else 1.0
        x[0] = min_forward_vel
        # 保持侧向和垂直方向的相对比例
        if forward_ratio > 1e-6:
            x[1:] = x[1:] * (min_forward_vel / (x[0] + 1e-6))
    
    command.velocity = x

    # manual speedup - 在起始位置增加前向速度
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
    
    # 调试输出
    if rospy.get_time() - int(rospy.get_time()) < 0.1:
        print(f'[USER_CODE] 最终速度命令 velocity = {command.velocity}, pos[0] = {state.pos[0]:.2f}')
    

    # creating debug images,
    # debugimg1 of the stabilized, cropped image with a velocity vector, and 
    # debugimg2 of the original image with the four points used for stabilization

    h, w = img2.shape
    arrow_start = (int(w/2), int(h/2))    
    arrow_end = (int(w/2-command.velocity[1]*(w/3)), int(h/2-command.velocity[2]*(h/3)))
    debugimg1 = cv2.arrowedLine( img2, arrow_start, arrow_end, (0, 0, 255), 10, )

    debugimg2 = orig_img.copy()

    return command, (debugimg1, debugimg2), hidden_state

# helper function for vectorized expert policy (method_id = 1)
def find_closest_zero_index(arr):
    center = np.array(arr.shape) // 2  # find the center point of the array
    dist_to_center = np.abs(np.indices(arr.shape) - center.reshape(-1, 1, 1)).sum(0)  # calculate distance to center for each element
    zero_indices = np.argwhere(arr == 0)  # find indices of all zero elements
    if len(zero_indices) == 0:
        return None  # if no zero elements, return None
    dist_to_zeros = dist_to_center[tuple(zero_indices.T)]  # get distances to center for zero elements
    min_dist_indices = np.argwhere(dist_to_zeros == dist_to_zeros.min()).flatten()  # find indices of zero elements with minimum distance to center
    chosen_index = np.random.choice(min_dist_indices)  # randomly choose one of the zero elements with minimum distance to center
    return tuple(zero_indices[chosen_index])  # return index tuple

def compute_command_state_based(state, obstacles, desiredVel, rl_policy=None, keyboard=False, keyboard_input=''):
    # print("Computing command based on obstacle information!")
    # print("Obstacles: ", obstacles)

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.yawrate = 0.0

    obst_dist_threshold = 8
    obst_inflate_factor = 0.6 #0.4#0.6
    method_id = 1 # 0 = old spiral method, 1 = new re-factored, 2 = constant
    if keyboard:
        import select
        method_id = 3

    # calculate an obstacle-free waypoint
    x_displacement = 8 #5
    grid_center_offset = 8
    grid_displacement = 0.5
    y_vals = np.arange(-grid_center_offset, grid_center_offset + grid_displacement, grid_displacement)
    num_wpts = y_vals.size

    start = time.time()

    # old expert
    if method_id == 0:

        wpts_2d = np.zeros((num_wpts, num_wpts, 2))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x, y]

        # the first layer of wpts_2d is actually the world y axis, the second is z axis
        # the third, the x axis, should all be +5m forward
        x_slice = x_displacement * np.ones((num_wpts, num_wpts))
        wpts_2d = np.concatenate((x_slice[:, :, None], wpts_2d), axis=2)

        # try spiraling outward again but just using bounds instead, and selecting blocks
        idx_midpt = num_wpts // 2
        curr_x = idx_midpt
        curr_y = idx_midpt
        x_bound = 1
        y_bound = -1
        wpt_idxs_2d = []
        count = 0
        while curr_x < num_wpts:
            if count % 4 == 0:
                x_bound = count / 4 + 1
            if (count - 1) % 4 == 0:
                y_bound = -((count - 1) / 4 + 1)

            if not count % 2:  # x-dir vector
                xvals = np.arange(
                    curr_x, idx_midpt + x_bound, -1 if x_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(np.repeat(int(curr_y), xvals.size), xvals)
                ]
                curr_x = idx_midpt + x_bound
                x_bound *= -1

            else:  # y-dir vector
                yvals = np.arange(
                    curr_y, idx_midpt + y_bound, -1 if y_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(yvals, np.repeat(int(curr_x), yvals.size))
                ]
                curr_y = idx_midpt + y_bound
                y_bound *= -1

            count += 1

        # iterate through waypoints, spiraling outwards from center
        for wpt_idx in wpt_idxs_2d:
            found_valid_pt = True
            # check if the current wpt is valid for all obstacles ahead of our current position
            for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                    found_valid_pt = False
                    break
            if found_valid_pt:
                break

        # CHECK AGAIN WITH OBSTACLE SCALE REDUCED TO .17
        if not found_valid_pt:
            print("[EXPERT] Didn't find a feasible path, Searching again with less inflation!")
            for wpt_idx in wpt_idxs_2d:
                found_valid_pt = True
                # check if the current wpt is valid for all obstacles ahead of our current position
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+0.17)):
                        found_valid_pt = False
                        break
                if found_valid_pt:
                    break
        
        # simplest controller: waypoint --PID--> linear velocity command
        yvel = 1.25 * (wpts_2d[wpt_idx][1])
        # x_scale_down_factor = (grid_center_offset - np.abs(yvel))/grid_center_offset
        xvel = max(desiredVel, 1 * (wpts_2d[wpt_idx][0]))
        zvel = 1.25 * wpts_2d[wpt_idx][2]

    # new expert
    elif method_id == 1:

        wpts_2d = np.zeros((num_wpts, num_wpts, 3))
        collisions = np.zeros((num_wpts, num_wpts))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x_displacement, x, y]
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    # print(f'wpt: {wpts_2d[yi, xi]} \t obst: {obst.position.x, obst.position.y, obst.position.z, obst.scale+obst_inflate_factor}')
                    if check_collision(((0, 0, 0), (wpts_2d[yi, xi])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                        collisions[yi, xi] = 1
                        break


        if collisions.sum() == collisions.size:
            print(f'[EXPERT] No collision-free path found')
            xvel = 0.5
            yvel = 0
            zvel = 0.25
        else:
            wpt_idx = find_closest_zero_index(collisions)
            wpt = wpts_2d[wpt_idx[0], wpt_idx[1]]

            # make the desired velocity vector of magnitude desiredVel
            wpt = (wpt / np.linalg.norm(wpt)) * desiredVel
            xvel = wpt[0]
            yvel = wpt[1]
            zvel = wpt[2]

    # just fly forward
    elif method_id == 2:

        xvel, yvel, zvel = (4.0, 0., 0.)

    elif method_id == 3:

        xvel, yvel, zvel = (2., 0., 0.)

        # print(f'[EXPERT] Keyboard input: {keyboard_input}')

        # Check if there is any keypress
        if keyboard_input == 'w':
            zvel = 1.0
        elif keyboard_input == 's':
            zvel = -1.0
        elif keyboard_input == 'a':
            yvel = 1.0
        elif keyboard_input == 'd':
            yvel = -1.0

        # norm the command vector up to desiredVel
        scaler = desiredVel/np.linalg.norm([xvel, yvel, zvel])
        xvel, yvel, zvel = (xvel*scaler, yvel*scaler, zvel*scaler)


    if time.time() - int(time.time()) < 0.1: # print this as infrequently as possible
        print(f'[EXPERT] Expert method {method_id} took {time.time() - start:.3f} seconds')

    command.velocity = [xvel, yvel, zvel]

    # recover altitude if too low
    if state.pos[2] < 2:
        command.velocity[2] = (2 - state.pos[2]) * 2

    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
        

    ################################################
    # !!! End !!!
    ###############################################

    return command
