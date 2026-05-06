#!/usr/bin/python3
import argparse

import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from envsim_msgs.msg import ObstacleArray

# from rl_example import load_rl_policy
from user_code import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState

import time
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
from copy import deepcopy
import cv2
import torch

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import *

# Branch model imports — each branch has its own model directory
_BRANCH_MODEL_DIRS = [
    opj(os.path.dirname(os.path.abspath(__file__)), '../../experiments/mamba_branches/branch_A_vmamba_lstm/models'),
    opj(os.path.dirname(os.path.abspath(__file__)), '../../experiments/mamba_branches/branch_B_mambavision_ssm/models'),
    opj(os.path.dirname(os.path.abspath(__file__)), '../../experiments/mamba_branches/branch_Bplus_mambavision_mamba3/models'),
    opj(os.path.dirname(os.path.abspath(__file__)), '../../experiments/mamba_branches/branch_C_cnn_mamba3/models'),
    opj(os.path.dirname(os.path.abspath(__file__)), '../../experiments/mamba_branches/branch_D_sth_mamba/models'),
    opj(os.path.dirname(os.path.abspath(__file__)), '../../experiments/mamba_branches/branch_E_decisionmamba/models'),
]
for _d in _BRANCH_MODEL_DIRS:
    if _d not in sys.path:
        sys.path.insert(0, _d)

try:
    from vmamba_lstm_model import VMambaLSTMNet, create_vmamba_lstm_model
    from mambavision_ssm_model import MambaVisionSSMNet, create_mambavision_ssm_model
    from cnn_mamba3_model import CNNMamba3Net, create_cnn_mamba3_model
    from sth_mamba_model import STHMambaNet, create_sth_mamba_model
    from decision_mamba_model import DecisionMambaNet, create_decision_mamba_model
    from bplus_model import BPlusModel
except ImportError as _e:
    print(f"[RUN_COMPETITION] Warning: branch model import failed: {_e}")

class AgilePilotNode:
    def __init__(self, vision_based=False, model_type=None, model_path=None, desVel=None, keyboard=False, seq_len=1):
        print("[RUN_COMPETITION] Initializing agile_pilot_node...")
        rospy.init_node("agile_pilot_node", anonymous=False)

        self.vision_based = vision_based
        self.rl_policy = None
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.keyboard = keyboard
        self.seq_len = seq_len
        self.frame_buffer = []

        quad_name = "kingfisher"

        self.init = 0
        self.col = None
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        self.last_valid_img = None #Image that will be logged
        data_log_format = {'timestamp':[],
                           'desired_vel':[],
                           'quat_1':[],
                           'quat_2':[],
                           'quat_3':[],
                           'quat_4':[],
                           'pos_x':[],
                           'pos_y':[],
                           'pos_z':[],
                           'vel_x':[],
                           'vel_y':[],
                           'vel_z':[],
                           'velcmd_x':[],
                           'velcmd_y':[],
                           'velcmd_z':[],
                           'ct_cmd':[],
                           'br_cmd_x':[],
                           'br_cmd_y':[],
                           'br_cmd_z':[],
                           'is_collide': [],
        } 
        self.data_log = pd.DataFrame(data_log_format) # store in the data frame
        self.count = 0 # counter for the csv
        
        # @NOTE: Dont log too fast, I have not tested that
        self.time_interval = .03 #Time interval for logging

        self.data_collection_xrange = [2, 60]

        # make the folder for the epoch
        self.folder = f"train_set/{int(time.time()*100)}"
        os.makedirs(self.folder, exist_ok=True)

        self.desiredVel = desVel #self.readVel("velocity.txt") #np.random.uniform(low=2.0, high=3.0)
        print()
        print(f"[RUN_COMPETITION] Desired velocity = {self.desiredVel}")
        print()

        # load trained model here (copied over from user_code.py)
        if model_path is not None:
            print(f"[RUN_COMPETITION] Model loading from {model_path} ...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if model_type == 'LSTMNet':
                self.model = LSTMNet().to(self.device).float()
            elif model_type == 'UNetLSTM':
                self.model = UNetConvLSTMNet().to(self.device).float()
            elif model_type == 'ConvNet':
                self.model = ConvNet().to(self.device).float()                
            elif model_type == 'ViT':
                self.model = ViT().to(self.device).float()
            elif model_type == 'ViTLSTM':
                # Use TeacherVITLSTM (input_size=517) — LSTMNetVIT (519) can't load the upstream checkpoint
                from model import TeacherVITLSTM
                self.model = TeacherVITLSTM().to(self.device).float()
            elif model_type == 'DroneMamba':
                self.model = DroneMamba(use_temporal_ssm=True, d_state=8).to(self.device).float()
                print(f"[RUN_COMPETITION] DroneMamba (SSM) loaded")
            elif model_type == 'VMambaLSTM':
                self.model = VMambaLSTMNet().to(self.device).float()
                print(f"[RUN_COMPETITION] Branch A — VMambaLSTMNet loaded")
            elif model_type == 'MambaVisionSSM':
                # Use the same config as training (from train_mamba_optimized.py lines 383-391)
                config = {
                    'mambavision_config': {
                        'in_channels': 1,
                        'stem_dim': 48,
                        'stage_dims': (64, 128, 192),
                        'depths': (2, 2, 2),
                        'd_state': 12,
                        'dropout': 0.3,
                        'output_dim': 512
                    },
                    'ssm_d_state': 16,
                    'ssm_hidden': 256,
                    'ssm_layers': 2,
                    'dropout': 0.3
                }
                self.model = create_mambavision_ssm_model(config).to(self.device).float()
                print(f"[RUN_COMPETITION] Branch B — MambaVisionSSMNet loaded (stem_dim=48, stage_dims=[64,128,192])")
            elif model_type == 'CNNMamba3':
                self.model = create_cnn_mamba3_model({'ssm_d_state': 16}).to(self.device).float()
                print(f"[RUN_COMPETITION] Branch C — CNNMamba3Net loaded (ssm_d_state=16)")
            elif model_type == 'BPlusModel':
                self.model = BPlusModel().to(self.device).float()
                print(f"[RUN_COMPETITION] Branch B+ — BPlusModel loaded (MambaVision + Mamba3 hybrid)")
            elif model_type == 'STHMamba':
                self.model = create_sth_mamba_model({}).to(self.device).float()
                print(f"[RUN_COMPETITION] Branch D — STHMambaNet loaded")
            elif model_type == 'DecisionMamba':
                self.model = create_decision_mamba_model({}).to(self.device).float()
                print(f"[RUN_COMPETITION] Branch E — DecisionMambaNet loaded")
            else:
                print(f'[RUN_COMPETITION] Invalid model_type {model_type}. Exiting.')
                exit()

            # Give full path if possible since the bash script runs from outside the folder
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

            # Remove '_orig_mod.' prefix from torch.compile() compiled models
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                print(f"[RUN_COMPETITION] Removed '_orig_mod.' prefix from compiled model weights")

            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = None

            print(f"[RUN_COMPETITION] Model loaded")
            time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0
        self.depth_im_threshold = 0.09

        self.curr_cmd = None

        # Logic subscribers
        self.start_sub = rospy.Subscriber(
            "/" + quad_name + "/start_navigation",
            Empty,
            self.start_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Observation subscribers
        self.odom_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/state",
            QuadState,
            self.state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/depth",
            Image,
            self.img_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles",
            ObstacleArray,
            self.obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.cmd_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/command",
            Command,
            self.cmd_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.keyboard_sub = rospy.Subscriber(
            "/keyboard_input",
            String,
            self.keyboard_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.rgb_img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/image",
            Image,
            self.rgb_callback,
            queue_size=1,
            tcp_nodelay=True,
        )


        # Command publishers
        self.cmd_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/feedthrough_command",
            Command,
            queue_size=1,
        )
        self.linvel_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/velocity_command",
            TwistStamped,
            queue_size=1,
        )
        self.debug_img1_pub = rospy.Publisher(
            "/debug_img1",
            Image,
            queue_size=1,
        )
        self.debug_img2_pub = rospy.Publisher(
            "/debug_img2",
            Image,
            queue_size=1,
        )
        self.depth_viz_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/unity/depth_viz",
            Image,
            queue_size=1,
        )
        self.vel_marker_pub = rospy.Publisher(
            "/debug/vel_marker",
            Marker,
            queue_size=1,
        )
        print("[RUN_COMPETITION] Initialization completed!")

        self.ctr = 0

        self.keyboard_input = ''
        self.got_keypress = 0.0
        self.rgb_img = None

    def rgb_callback(self, img):
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    def cmd_callback(self, msg):
        self.curr_cmd = msg

    def keyboard_callback(self, msg):
        self.got_keypress = rospy.Time().now().to_sec()
        self.keyboard_input = msg.data

    def readVel(self,file):
        with open(file,"r") as f:
            x = f.readlines()
            for i in range(len(x)):
                if i == 0:
                    return float(x[i].split("\n")[0])

    def img_callback(self, img_data):
        self.ctr += 1
        self.prevImg = deepcopy(self.last_valid_img)
        try:
            img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        except Exception as e:
            print(f"[img_callback] cv_bridge error: {e}, using fallback")
            img = np.zeros((480, 640), dtype=np.float32)

        if img.size == 0 or img.shape[0] == 0:
            img = np.zeros((480, 640), dtype=np.float32)

        if self.last_valid_img is None:
            self.prevImg = np.ones((480, 640), dtype=np.float32) * 10.0

        img = np.clip(img/self.depth_im_threshold, 0, 1)

        if self.prevImg is None:
            self.prevImg = img

        if img.size > 0 and img.shape[0] > 0:
            try:
                self.last_valid_img = deepcopy(img) if img.min() > 0.0 else self.last_valid_img
            except:
                self.last_valid_img = np.ones((480, 640), dtype=np.float32) * 10.0
        
        
        
        
        

        if not self.vision_based:
            return
        
        if self.state is None:
            return
        
        # Buffer frames for multi-step inference
        if self.seq_len > 1:
            self.frame_buffer.append(img)
            if len(self.frame_buffer) < self.seq_len:
                return  # wait for buffer to fill
            # Slide window: keep most recent seq_len frames
            if len(self.frame_buffer) > self.seq_len:
                self.frame_buffer.pop(0)
            infer_img = np.stack(self.frame_buffer, axis=0)  # (S, H, W)
        else:
            infer_img = img
        
        start_compute_time = time.time()

        command, (debug_img1, debug_img2), self.model_hidden_state = compute_command_vision_based(
            self.state, infer_img, self.prevImg, self.desiredVel,
            self.model, self.model_hidden_state, seq_len=self.seq_len)

        # publish debug images
        self.debug_img1_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="passthrough"))
        self.debug_img2_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="passthrough"))

        # publish depth visualization for RViz
        depth_viz = (img * 255).astype(np.uint8)
        self.depth_viz_pub.publish(self.cv_bridge.cv2_to_imgmsg(depth_viz, encoding="mono8"))

        # publish velocity marker for RViz
        self.publish_velocity_marker(command)

        if self.ctr % 30 == 0:
            print(f'[RUN_COMPETITION] compute_command_vision_based took {time.time() - start_compute_time} seconds')

        self.publish_command(command)
        # print(f'[RUN_COMPETITION] output: {command.velocity}')

        if self.state.pos[0] < 0.1:
            self.start_time = command.t

        if self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        #if we exceed the time interval then save the data
        if (self.state.t - self.t1 > self.time_interval or self.t1==0) and self.state.pos[0] < 63:
            #reset the time flag
            self.t1 = self.state.t

            # Get the current time stamp - instant
            timestamp = round(
                self.state.t, 3
            )  # If you need more hz, you might need to modify this round

            # Save the image by the name of that instant
            cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img*255).astype(np.uint8))

            # Get the collision flag
            if self.col is None:
                self.col = 0
            # Append the data frame
            # @TODO: This needs to be managed better if the number of datapoints exceeds 10,000
            self.data_log.loc[len(self.data_log)] = [
                timestamp,
                self.desiredVel,
                self.state.att[0],
                self.state.att[1],
                self.state.att[2],
                self.state.att[3],
                self.state.pos[0],
                self.state.pos[1],
                self.state.pos[2],
                self.state.vel[0],
                self.state.vel[1],
                self.state.vel[2],
                command.velocity[0],
                command.velocity[1],
                command.velocity[2],
                self.curr_cmd.collective_thrust,
                self.curr_cmd.bodyrates.x,
                self.curr_cmd.bodyrates.y,
                self.curr_cmd.bodyrates.z,
                self.col,
            ]

            # Counter flag for saving the data frame
            self.count += 1

        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 5 == 0:
            self.data_log.to_csv(self.folder + "/data.csv")

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)

    def obstacle_callback(self, obs_data):
        if self.state is None:
            return
        self.col = self.if_collide(obs_data.obstacles[0])
        if self.vision_based:
            return
        if self.rgb_img is None:
            print("no rgb image yet")
            return

        # try:
        #     self.desiredVel = self.readVel("velocity.txt") #Changed some thing
        # except:
        #     pass
        # usable keypress?
        if rospy.Time().now().to_sec() - self.got_keypress > 0.1:
            self.keyboard_input = ''

        command = compute_command_state_based(
            state=self.state,
            obstacles=obs_data,
            desiredVel=self.desiredVel,
            rl_policy=self.rl_policy,
            keyboard=self.keyboard,
            keyboard_input=self.keyboard_input,
        )
        self.publish_command(command)

        if self.state.pos[0] < 0.1:
            self.start_time = command.t
        if self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        # if we exceed the time interval then save the data
        if (self.state.t - self.t1 > self.time_interval or self.t1 == 0 or self.col) and (self.state.pos[2] > 2.95 or self.init == 1):
            
            self.init = 1

            if self.state.pos[0] > self.data_collection_xrange[0] and self.state.pos[0] < self.data_collection_xrange[1]:

                # reset the time flag
                self.t1 = self.state.t

                # Get the current time stamp
                timestamp = round(self.state.t, 3)  # If you need more hz, you might need to modify this round

                # Save the image by the name of that instant
                # np.save(self.folder + f"/im_{timestamp}", self.last_valid_img)
                cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img*255).astype(np.uint8))
                cv2.imwrite(f"{self.folder}/{str(timestamp)}_rgb.png", (self.rgb_img*255).astype(np.uint8))

                # Get the collision flag
                col = self.if_collide(obs_data.obstacles[0])
                # Append the data frame
                # @TODO: This needs to be managed better if the number of datapoints exceeds 10,000
                self.data_log.loc[len(self.data_log)] = [
                    timestamp,
                    self.desiredVel,
                    self.state.att[0],
                    self.state.att[1],
                    self.state.att[2],
                    self.state.att[3],
                    self.state.pos[0],
                    self.state.pos[1],
                    self.state.pos[2],
                    self.state.vel[0],
                    self.state.vel[1],
                    self.state.vel[2],
                    command.velocity[0],
                    command.velocity[1],
                    command.velocity[2],
                    self.curr_cmd.collective_thrust,
                    self.curr_cmd.bodyrates.x,
                    self.curr_cmd.bodyrates.y,
                    self.curr_cmd.bodyrates.z,
                    self.col,
                ]

                # Counter flag for saving the data frame
                self.count += 1

        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 2 == 0 and self.count != 0 or abs(self.state.pos[0] - 20) < 1:
            self.data_log.to_csv(self.folder + "/data.csv")

    def if_collide(self, obs):
        """
        Borrowed and modified from evaluation_node
        """

        dist = np.linalg.norm(
            np.array([obs.position.x, obs.position.y, obs.position.z])
        )
        margin = dist - obs.scale
        # Ground hit condition
        if margin < 0 or self.state.pos[2] <= 0.01:
            hit_obstacle = True
        else:
            hit_obstacle = False

        return hit_obstacle

    def publish_velocity_marker(self, command):
        """Publish velocity command as arrow marker for RViz visualization"""
        if self.state is None:
            return

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "velocity"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow starts at drone position
        marker.pose.position.x = self.state.pos[0]
        marker.pose.position.y = self.state.pos[1]
        marker.pose.position.z = self.state.pos[2]

        # Arrow direction from velocity vector
        vel_norm = np.linalg.norm(command.velocity)
        if vel_norm > 0.01:
            # Normalize and scale for visibility
            vel_dir = command.velocity / vel_norm
            # Convert to quaternion (arrow points along x-axis by default)
            yaw = np.arctan2(vel_dir[1], vel_dir[0])
            pitch = np.arcsin(-vel_dir[2])
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = np.sin(pitch/2)
            marker.pose.orientation.z = np.sin(yaw/2) * np.cos(pitch/2)
            marker.pose.orientation.w = np.cos(yaw/2) * np.cos(pitch/2)
        else:
            marker.pose.orientation.w = 1.0

        # Arrow size (length proportional to velocity magnitude)
        marker.scale.x = min(vel_norm * 0.5, 3.0)  # shaft length
        marker.scale.y = 0.1  # shaft diameter
        marker.scale.z = 0.15  # head diameter

        # Color: green for forward, red for backward
        marker.color.r = 0.0 if command.velocity[0] > 0 else 1.0
        marker.color.g = 1.0 if command.velocity[0] > 0 else 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0.2)

        self.vel_marker_pub.publish(marker)

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                print(f"[RUN_COMPETITION] Published velocity: {command.velocity}")
                return
            else:
                print(f"[RUN_COMPETITION] NOT publishing (publish_commands=False), velocity: {command.velocity}")
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("[RUN_COMPETITION] Start publishing commands!")
        self.publish_commands = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot.")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to model checkpoint')
    parser.add_argument('--des_vel', type=float, default=None, help='desired velocity for quadrotor')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")
    parser.add_argument('--seq-len', type=int, default=1, help='Number of frames to buffer for multi-step inference (default: 1 = single-step)')

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, model_type=args.model_type, model_path=args.model_path, desVel=args.des_vel, keyboard=args.keyboard, seq_len=args.seq_len)
    rospy.spin()
