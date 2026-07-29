#!/usr/bin/env python3

"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@brief: This script runs a rosnode that subscribes to depth images, processes them with a learned model, and outputs velocity commands.
"""

import rospy
import numpy as np
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import sys, os
import time
import torch

DEPTHFLY_PATH='/home/anish/vitfly_ws/src/vitfly/depthfly/'

sys.path.append(DEPTHFLY_PATH+'../models')
from model import *

class DepthInferenceNode:
    def __init__(self):
        rospy.init_node('inference_node') #, anonymous=True)

        self.images = []
        self.evframe = None
        self.cv_bridge = CvBridge()

        ### INITIALIZE MODEL

        self.desired_velocity = 4.0
        self.model_type = 'ViTLSTM'
        self.model_path = '/home/anish/vitfly_ws/src/vitfly/models/ViTLSTM_model.pth'

        # define our model
        if self.model_path is not None:

            print(f"[DEPTHFLY RUN] Model loading from {self.model_path} ...")
            self.device = torch.device("cpu")
            if self.model_type == 'LSTMNet':
                self.model = LSTMNet().to(self.device).float()
            elif self.model_type == 'UNetLSTM':
                self.model = UNetConvLSTMNet().to(self.device).float()
            elif self.model_type == 'ConvNet':
                self.model = ConvNet().to(self.device).float()                
            elif self.model_type == 'ViT':
                self.model = ViT().to(self.device).float()
            elif self.model_type == 'ViTLSTM':
                self.model = LSTMNetVIT().to(self.device).float()                
            else:
                print(f'[DEPTHFLY RUN] Invalid self.model_type {self.model_type}. Exiting.')
                exit()

            # Give full path if possible since the bash script runs from outside the folder
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = None

            print(f"[DEPTHFLY RUN] Model loaded")
            time.sleep(2)

        else:
            print(f"[DEPTHFLY RUN] No model path given, so exiting.")
            exit()

        ### SUBSCRIBERS

        self.depth_image_raw = None
        self.image_subscriber = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback, queue_size=1)

        # subscriber to odometry
        self.odom = None
        self.odom_subscriber = rospy.Subscriber('/robot/odom', Odometry, self.odom_callback, queue_size=1)

        self.last_trigger_t = 0.0
        self.first_trigger_t = None
        self.trigger_subscriber = rospy.Subscriber('/trigger', Empty, self.trigger_callback, queue_size=1)

        ### PUBLISHERS

        # debug publisher
        self.dbg_im = None
        self.dbg_im_publisher = rospy.Publisher('/output/dbg_im', Image, queue_size=1)

        self.pred_vel = None
        self.pred_vel_publisher = rospy.Publisher('/output/pred_vel', TwistStamped, queue_size=1)

        self.vel_msg = TwistStamped()
        self.vel_msg.header.stamp = rospy.Time.now()
        self.vel_msg.twist.linear.x = 0.0
        self.vel_msg.twist.linear.y = 0.0
        self.vel_msg.twist.linear.z = 0.0

        self.vel_cmd_publisher = rospy.Publisher('/robot/cmd_vel', TwistStamped, queue_size=1)

        self.rate = rospy.Rate(30) # 5Hz
 
    def trigger_callback(self, msg):
        if self.first_trigger_t is None:
            self.first_trigger_t = rospy.Time().now().to_sec()
        self.last_trigger_t = rospy.Time().now().to_sec()

    def odom_callback(self, msg):
        self.odom = msg

    def run_model(self):

        # prepare inputs to network

        # resize input to 60x90
        input_frame = torch.nn.functional.interpolate(self.depth_im.view(1, 1, self.depth_im.shape[0], self.depth_im.shape[1]), size=(60, 90), mode='bilinear', align_corners=False).to(self.device).float()
        # input_frame = self.depth_im.view(1, 1, 60, 90).to(self.device).float()
        desvel = torch.Tensor([[self.desired_velocity]]).to(self.device)
        if self.odom is not None:
           quad_att = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]
        else:
            quad_att = None

        if 'LSTM' in self.model_type:
            inputs_to_model = [input_frame, desvel, quad_att, self.model_hidden_state]
        else:
            inputs_to_model = [input_frame, desvel, quad_att]

        # st_modelfwd_time = time.time()
        with torch.no_grad():
            x_vel, self.model_hidden_state = self.model(inputs_to_model)
        # print(f"model inference took {time.time()-st_modelfwd_time:.3f} seconds")

        self.pred_vel = x_vel.cpu().detach().numpy().squeeze() # keeping this 0-1
        
    def publish_dbg_im(self):

        # convert to uint8
        dbg_im_uint8 = (self.dbg_im * 255).astype(np.uint8)
        #pred_depth_uint8 = (np.abs(self.evframe)/np.abs(self.evframe).max() * 255).astype(np.uint8)

        # convert to rosmsg
        dbg_im_rosmsg = self.cv_bridge.cv2_to_imgmsg(dbg_im_uint8) #, encoding="mono8")

        # publish
        self.dbg_im_publisher.publish(dbg_im_rosmsg)

    def publish_pred_vel(self):

        self.pred_vel *= self.desired_velocity

        self.vel_msg = TwistStamped()
        self.vel_msg.header.stamp = rospy.Time.now()
        self.vel_msg.twist.linear.x = self.pred_vel[0]
        self.vel_msg.twist.linear.y = self.pred_vel[1]
        if self.odom is not None:
            self.vel_msg.twist.linear.z = 1.5 * (1.0 - self.odom.pose.pose.position.z) # self.pred_vel[2]
        else:
            self.vel_msg.twist.linear.z = 0.0
        self.pred_vel_publisher.publish(self.vel_msg)

    def depth_callback(self, msg):

        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Convert OpenCV image to numpy array
        self.depth_image_raw = np.array(cv_image, dtype=np.float32)

    def process(self):

        # NOTE, might be realsense d435 camera-specific
        depth_scale_factor = 10e3 # scale down and clip depth image (mm)

        if self.depth_image_raw is not None:

            self.depth_im = torch.from_numpy(self.depth_image_raw)
            # self.depth_im = torch.clip(self.depth_im/depth_scale_factor, 0.0, 1.0)

            # NOTE, with a realsense d435 camera, the following adjustments may improve performance
            self.depth_im = torch.clip(self.depth_im/depth_scale_factor, 0.0, 1.0/0.8) * 0.8
            # make 0.0 points near 1.0
            self.depth_im[self.depth_im == 0.0] = 0.8

            self.dbg_im = self.depth_im.numpy()

            self.run_model()

            # write arrow
            if self.pred_vel is not None:
                im_w, im_h = self.depth_im.shape[1], self.depth_im.shape[0]
                arrow_start = (im_w//2, im_h//2)
                arrow_end = (int(im_w/2-self.pred_vel[1]*(im_w/3)), int(im_h/2-self.pred_vel[2]*(im_h/3)))

                # self.dbg_im = cv2.arrowedLine( self.dbg_im, arrow_start, arrow_end, (0, 0, 255), im_h//60, tipLength=0.2)

            if self.pred_vel is not None:
                self.publish_pred_vel()
            
            if self.dbg_im is not None:
                self.publish_dbg_im()

    def run(self):
        while not rospy.is_shutdown():
            
            self.process()

            if rospy.Time().now().to_sec() - self.last_trigger_t < 0.1:
                #print("commanding!")

                # manual, discretized ramp-up; in first second of commands, reduce fwd/dodging vel by /2.0
                ramp_duration = 2.0
                if rospy.Time().now().to_sec() - self.first_trigger_t < ramp_duration:

                    ramp_time = rospy.Time().now().to_sec() - self.first_trigger_t
                    ramp_scaler = ramp_time / ramp_duration

                    self.vel_msg.twist.linear.x *= ramp_scaler
                    self.vel_msg.twist.linear.y *= ramp_scaler

                    self.vel_msg.twist.linear.x = max(min(1.0 + self.vel_msg.twist.linear.x, 7.0), 0.0)
                
                self.vel_cmd_publisher.publish(self.vel_msg)

            else:

                if self.last_trigger_t > 0.0:
                    
                    self.vel_msg.twist.linear.x = 0.0
                    self.vel_msg.twist.linear.y = 0.0
                    self.vel_msg.twist.linear.z = 0.0
                    self.vel_cmd_publisher.publish(self.vel_msg)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        depth_inference_node = DepthInferenceNode()
        depth_inference_node.run()
    except rospy.ROSInterruptException:
        pass
