#!/usr/bin/env python


import time
import rospy
import csv
import os
import numpy as np
import json
import math

from pycrazyswarm import *
from position_ctl_m import PositionController
from quadrotor import quadrotor
from helper import quat2euler, rad2deg, deg2rad
from trajectory_generator import TrajectoryGenerator2DPeriodicMotion, TrajectoryGenerator3DPeriodicMotion
from crazyswarm.msg import StateVector, Command
from geometry_msgs.msg import TransformStamped
from utils import DataVarIndex, Status
#from vicon_bridge.msg import Command


class DataLogger:
    """A class that logs the recorded data to a csv file using the DataVarIndex."""

    def __init__(self, filename):
        self.filename = filename
        self.create_csv()

    def create_csv(self):
        # Create the csv file
        with open(self.filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow([var.name for var in DataVarIndex])

    def log_data(self, data):
        """Log the data to the csv file."""
        # Make sure that the data has the correct length
        assert len(data) == len(DataVarIndex)

        # Log the data
        with open(self.filename, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(data)


class ViconWatcher:
    def __init__(self, cf_id_dec = 'cf01'):
        self.topic = f'/vicon/{cf_id_dec}/{cf_id_dec}'
        self.vicon_sub = rospy.Subscriber(self.topic, TransformStamped, self.vicon_callback)
        self.pos = None
        self.rpy = None

    def vicon_callback(self, data):
        self.child_frame_id = data.child_frame_id
        self.pos = np.array([
            data.transform.translation.x,
            data.transform.translation.y,
            data.transform.translation.z,
        ])
        rpy = quat2euler(
            data.transform.rotation.x,
            data.transform.rotation.y,
            data.transform.rotation.z,
            data.transform.rotation.w,
        )
        self.rpy = np.array(rpy)


class StateEstimator:
    def __init__(self):
        self.topic = 'estimated_state'
        print(self.topic)
        self.state_estimation_sub = rospy.Subscriber(self.topic, StateVector, self.estimator_callback)
        
        self.pos = None
        self.rpy = None
        self.vel = None
        self.omega = None

    def estimator_callback(self, data):
        self.pos = np.array(data.pos)
        rpy = quat2euler(
            data.quat[0],
            data.quat[1],
            data.quat[2],
            data.quat[3],
        )
        self.rpy = np.array(rpy)
        self.vel = np.array(data.vel)
        #self.omega = np.array(data.omega_b)


class QuadMotion:

    def __init__(self, state_estimator=None, vicon=None, control_freq=30, grad_start=True, verbose=False, log_data=False, filename=None, sim=False):

        self.control_freq = control_freq
        self.dt = 1.0 / control_freq

        # Read parameters file for the controller gains
        script_dir = os.path.dirname(__file__)
        param_file = os.path.join(script_dir, "parameters.json")
        with open(param_file) as f:
            params = json.load(f)
            self.MPC_N = params['MPC_solver']['N'] # Prediction Horizon

        self.quad = quadrotor()
        self.params = self.quad.params

        crazyswarm_args = ""
        self.sim = sim
        if self.sim: 
            crazyswarm_args = "--sim --vis null"
        self.swarm = Crazyswarm(args=crazyswarm_args)
        self.timeHelper = self.swarm.timeHelper
        # Only controls one crazyflie
        self.cf = self.swarm.allcfs.crazyflies[0]
        self.posCtrl = PositionController(self.quad, self.control_freq)

        # Initialize publisher for control commands
        self.controller_pub = rospy.Publisher('/controller_command', Command, queue_size=1)
        
        # Check availability of input-communication tunnel
        self.check_input_tunnel_connection_availability()
        
        # Initialize the control command
        self.controller_command = Command()

        # Check whether estimator start publishing estimated state (firstly raw data)
        if state_estimator:
            self.state_estimator = state_estimator
            
            self.check_estimator_availability()
            
            self.init_position = self.state_estimator.pos
        else:
            raise ValueError("State estimator object is required.")
        
        self.vicon = vicon
                
        self.verbose = verbose
        self.log_data = log_data
        if self.log_data:
            assert filename is not None
            self.filename = filename
            self.log_data_init()

        self.prev_t = None
        self.sleep_offset_mean = 0.0
        self.sleep_offset_counter = 0


        "----------for test----------"
        self.counter = 0
        self.delta_t_0_ave = 0
        self.delta_t_0_var = 0
        self.delta_t_1_ave = 0
        self.delta_t_1_var = 0
        self.delta_t_2_ave = 0
        self.delta_t_2_var = 0
        self.delta_t_sum_ave = 0
        self.delta_t_sum_var = 0
        "----------for test----------"
        
        # Define counter parameter for gradually starting
        self.grad_start = grad_start
        if self.grad_start:
            self.track_traj_counter = 0 # set value = 0 to switch on
        else:
            self.track_traj_counter = 31 # set value >= 31 to switch off
        


    def log_data_init(self):
        """ Initialize the data logger. """
        self.data_logger = DataLogger(self.filename)

    def check_input_tunnel_connection_availability(self):
        timeout = 100
        while self.controller_pub.get_num_connections() == 0:
            print('Waiting for subscriber of command in state estimator...')
            timeout -= 1
            time.sleep(1)
            if not timeout:
                raise TimeoutError(f'No connection tunnel for control command founded.')
        print('Connection tunnel for control command founded.')

    def check_estimator_availability(self):
        timeout = 100
        while self.state_estimator.pos is None or self.state_estimator.rpy is None:

            print('Waiting for state estimator...')
            timeout -= 1
            time.sleep(1)
            if not timeout:
                raise TimeoutError(f'No messages received from {self.state_estimator.topic}.')
            
            t = time.time() 
            self.controller_command.header.stamp.secs = int(t)
            self.controller_command.header.stamp.nsecs = int((t - self.controller_command.header.stamp.secs) * 1e9)
            self.controller_command.CMD_ROLL = 0
            self.controller_command.CMD_PITCH = 0
            self.controller_command.CMD_YAW = 0
            self.controller_command.CMD_PWM = 0.0209
            # Publish the control input
            self.controller_pub.publish(self.controller_command)

        print('State estimator found.')


    def idle(self):
        """ Initialize the cmdVel command. Otherwise, the drone will not move."""
        for i in range(10):
            self.cf.cmdVel(0.0, 0.0, 0.0, 0)
            self.timeHelper.sleepForRate(self.control_freq)

    def delta_t(self):
        """ Return the time difference between the current and previous time."""
        t = time.time()
        if self.prev_t is None:
            return self.dt
        dt = t - self.prev_t
        return dt









    '''basic behavior'''

    def pos_control(self, pos, rpy, vel, target_pos_arr, target_vel_arr, target_yaw_arr, status):
        """ Position control of the drone.

        Args:
            pos (np.array): current position of the drone.
            rpy (np.array): current roll, pitch, yaw angles of the drone in radians.
            target_pos (np.array): target position of the drone.
            target_vel (np.array): target velocity of the drone.
            target_yaw (float): target yaw angle of the drone in radians. # used in formula 20

        """

        #dt = self.delta_t()
        #dt = self.dt




        "----------for test----------"
        self.counter += 1
        t1 = time.time()
        if self.prev_t:
            delta_t_0 = t1 - self.prev_t
            self.delta_t_0_ave_old = self.delta_t_0_ave
            self.delta_t_0_ave += (delta_t_0 - self.delta_t_0_ave_old) / self.counter
            self.delta_t_0_var = (self.delta_t_0_var * (self.counter-2) + (delta_t_0 - self.delta_t_0_ave_old) * (delta_t_0 - self.delta_t_0_ave)) / self.counter
            rospy.loginfo("ave time since start of last cycle (inkl. sleep): %f. " %self.delta_t_0_ave)
            rospy.loginfo("var of time since start of last cycle (inkl. sleep): %f. " %math.sqrt(self.delta_t_0_var))
        "----------for test----------"





        pwm, euler, state_predicted = self.posCtrl.compute_action(pos, rpy, vel, target_pos_arr, target_vel_arr, target_yaw_arr)

        t = time.time() 
        # Update the command data to be published
        self.controller_command.header.stamp.secs = int(t)
        self.controller_command.header.stamp.nsecs = int((t - self.controller_command.header.stamp.secs) * 1e9)
        self.controller_command.CMD_ROLL = euler[0]
        self.controller_command.CMD_PITCH = euler[1]
        self.controller_command.CMD_YAW = euler[2]
        self.controller_command.CMD_PWM = pwm
        # Publish the control input 
        self.controller_pub.publish(self.controller_command)





        "----------for test----------"
        t2 = time.time()
        delta_t_1 = t2 - t1
        self.delta_t_1_ave_old = self.delta_t_1_ave
        self.delta_t_1_ave += (delta_t_1 - self.delta_t_1_ave_old) / self.counter
        self.delta_t_1_var = (self.delta_t_1_var * (self.counter-2) + (delta_t_1 - self.delta_t_1_ave_old) * (delta_t_1 - self.delta_t_1_ave)) / self.counter
        rospy.loginfo("ave cycle time for period 1 of controller: %f. " %self.delta_t_1_ave)
        rospy.loginfo("var of cycle time for period 1 of controller: %f. " %math.sqrt(self.delta_t_1_var))
        "----------for test----------"





        # Realise the command
        euler_deg = rad2deg(euler)

        self.cf.cmdVel(euler_deg[0], euler_deg[1], euler_deg[2], pwm)

        if self.log_data:
            data = [None] * len(DataVarIndex)
            data[DataVarIndex.TIME] = t
            data[DataVarIndex.POS_X] = pos[0]
            data[DataVarIndex.POS_Y] = pos[1]
            data[DataVarIndex.POS_Z] = pos[2]
            data[DataVarIndex.ROLL] = rpy[0]
            data[DataVarIndex.PITCH] = rpy[1]
            data[DataVarIndex.YAW] = rpy[2]
            data[DataVarIndex.VEL_X] = vel[0]
            data[DataVarIndex.VEL_Y] = vel[1]
            data[DataVarIndex.VEL_Z] = vel[2]
            data[DataVarIndex.CMD_ROLL] = deg2rad(euler_deg[0])
            data[DataVarIndex.CMD_PITCH] = deg2rad(euler_deg[1])
            data[DataVarIndex.CMD_YAW] = deg2rad(euler_deg[2])
            data[DataVarIndex.CMD_THRUST] = pwm
            data[DataVarIndex.DES_POS_X] = target_pos_arr[0, 0]
            data[DataVarIndex.DES_POS_Y] = target_pos_arr[0, 1]
            data[DataVarIndex.DES_POS_Z] = target_pos_arr[0, 2]
            data[DataVarIndex.DES_YAW] = float(target_yaw_arr[0])  # make target_yaw a float
            data[DataVarIndex.DES_VEL_X] = target_vel_arr[0, 0]
            data[DataVarIndex.DES_VEL_Y] = target_vel_arr[0, 1]
            data[DataVarIndex.DES_VEL_Z] = target_vel_arr[0, 2]
            data[DataVarIndex.STATUS] = status.name
            if self.vicon:
                data[DataVarIndex.VICON_POS_X] = self.vicon.pos[0]
                data[DataVarIndex.VICON_POS_Y] = self.vicon.pos[1]
                data[DataVarIndex.VICON_POS_Z] = self.vicon.pos[2]
                data[DataVarIndex.VICON_ROLL] = self.vicon.rpy[0]
                data[DataVarIndex.VICON_PITCH] = self.vicon.rpy[1]
                data[DataVarIndex.VICON_YAW] = self.vicon.rpy[2]
            for i in range(self.posCtrl.MPC_N + 1): # 0 ~ self.posCtrl.MPC_N
                pre_index = DataVarIndex.x_0 + i
                data[pre_index] = state_predicted[i]

            self.data_logger.log_data(data)




        "----------for test----------"
        t3 = time.time()
        delta_t_2 = t3 - t2
        self.delta_t_2_ave_old = self.delta_t_2_ave
        self.delta_t_2_ave += (delta_t_2 - self.delta_t_2_ave_old) / self.counter
        self.delta_t_2_var = (self.delta_t_2_var * (self.counter-2) + (delta_t_2 - self.delta_t_2_ave_old) * (delta_t_2 - self.delta_t_2_ave)) / self.counter
        rospy.loginfo("ave cycle time for period 2 of controller: %f. " %self.delta_t_2_ave)
        rospy.loginfo("var of cycle time for period 2 of controller: %f. " %math.sqrt(self.delta_t_2_var))
        "----------for test----------"
        delta_t_sum = t3 - t1
        self.delta_t_sum_ave_old = self.delta_t_sum_ave
        self.delta_t_sum_ave += (delta_t_sum - self.delta_t_sum_ave_old) / self.counter
        self.delta_t_sum_var = (self.delta_t_sum_var * (self.counter-2) + (delta_t_sum - self.delta_t_sum_ave_old) * (delta_t_sum - self.delta_t_sum_ave)) / self.counter
        rospy.loginfo("ave cycle time of controller: %f. " %self.delta_t_sum_ave)
        rospy.loginfo("var of cycle time of controller: %f. " %math.sqrt(self.delta_t_sum_var))
        "----------for test----------"






        if self.sim:
            #rospy.sleep(self.dt)
            # adjust sleep time to match control frequency
            sleep_time = self.dt
            if self.prev_t is not None:
                sleep_offset = t - self.prev_t - self.dt
                #print("Sleep offset: %.4f" % sleep_offset)

                # Use the mean of the sleep offset to adjust the sleep time
                sleep_offset_sum = self.sleep_offset_mean * self.sleep_offset_counter
                self.sleep_offset_counter += 1
                self.sleep_offset_mean = (sleep_offset_sum + sleep_offset) / self.sleep_offset_counter
                print("Sleep offset: %.4f" % self.sleep_offset_mean)
                if sleep_offset >= 1e-5:
                    #sleep_time -= sleep_offset
                    sleep_time -= self.sleep_offset_mean
            rospy.sleep(sleep_time)
            self.prev_t = t
        else:
            self.timeHelper.sleepForRate(self.control_freq)   # 0.0163s记时的开始与终止


        if self.verbose:
            rpy_deg = rad2deg(rpy)
            print("Target pos %.4f, %.4f, %.4f" % (target_pos_arr[0, 0], target_pos_arr[0, 1], target_pos_arr[0, 2]))
            print("Measured: roll %.4f, pitch %.4f, yaw %.4f" % (rpy_deg[0], rpy_deg[1], rpy_deg[2]))
            print("CMD: pwm %.4f, roll %.4f deg, pitch %.4f deg, yaw %.4f deg" % (pwm, euler_deg[0], euler_deg[1], euler_deg[2]))
        print("time %.2f" % t)







    '''lower level behavior'''

    def vertical(self, velocity=0.3, height=0.5, target_yaw_deg=0.0, status=Status.VERTICAL):
        """ Move the drone vertically up or down.
        
        Args:
            duration (float): duration of the vertical motion in seconds.
            height (float): height to move to in meters.
            target_yaw_deg (float): target yaw angle in degrees.
            status (Status): status of the drone.
            
        """
        init_pos = self.state_estimator.pos + np.zeros(3,)
        delta_height_total = height - init_pos[2]
        num_steps = int(abs(delta_height_total) / velocity * self.control_freq)
        delta_height = (height - init_pos[2]) / num_steps
        target_vel = np.array([0.0, 0.0, delta_height / self.dt])
        target_vel_arr = np.tile(target_vel, (self.MPC_N + 1, 1))
        target_yaw = deg2rad(target_yaw_deg)
        target_yaw_arr = np.tile(target_yaw, (self.MPC_N + 1, 1))

        for i in range(num_steps): # openloop control
            rpy = self.state_estimator.rpy
            pos = self.state_estimator.pos
            vel = self.state_estimator.vel
            target_pos = init_pos + np.array([0.0, 0.0, i * delta_height])
            target_pos_arr = np.tile(target_pos, (self.MPC_N + 1, 1))
            self.pos_control(pos, rpy, vel, target_pos_arr, target_vel_arr, target_yaw_arr, status=status)

    def horizontal(self, velocity=0.3, target_x=0.0, target_y=0.0, target_yaw_deg=0.0, status=Status.HORIZONTAL):
        """ Move the drone horizontally in the x-y plane.
        
        Args:
            duration (float): duration of the horizontal motion in seconds.
            target_x (float): target x position in meters.
            target_y (float): target y position in meters.
            target_yaw_deg (float): target yaw angle in degrees.
            status (Status): status of the drone.
            
        """
        init_pos = self.state_estimator.pos + np.zeros(3,)
        delta_x_total = target_x - init_pos[0]
        delta_y_total = target_y - init_pos[1]
        distance = np.sqrt(delta_x_total ** 2 + delta_y_total ** 2)
        num_steps = int(distance / velocity * self.control_freq) 
        if num_steps > 0:       
            delta_x = (target_x - init_pos[0]) / num_steps
            delta_y = (target_y - init_pos[1]) / num_steps
            target_vel = np.array([delta_x / self.dt, delta_y / self.dt, 0.0])
            target_vel_arr = np.tile(target_vel, (self.MPC_N + 1, 1))
            target_yaw = deg2rad(target_yaw_deg)
            target_yaw_arr = np.tile(target_yaw, (self.MPC_N + 1, 1))

            for i in range(num_steps):
                rpy = self.state_estimator.rpy
                pos = self.state_estimator.pos
                vel = self.state_estimator.vel
                target_pos = init_pos + np.array([i * delta_x, i * delta_y, 0.0])
                target_pos_arr = np.tile(target_pos, (self.MPC_N + 1, 1))

                self.pos_control(pos, rpy, vel, target_pos_arr, target_vel_arr, target_yaw_arr, status=status)

    def interpolate_vel(self, duration=2.0, target_vel=np.array([0.0, 0.0, 0.0]), target_yaw_deg=0.0, status=Status.INTERPOLATE):
        """ Interpolate the velocity of the drone.
        
        Args:
            duration (float): duration of the interpolation in seconds.
            target_vel (np.array): target velocity in m/s.
            target_yaw_deg (float): target yaw angle in degrees.
            status (Status): status of the drone.
            
        """
        num_steps = int(duration * self.control_freq)
        target_yaw = deg2rad(target_yaw_deg)

        for i in range(num_steps):
            alpha = i / num_steps
            rpy = self.state_estimator.rpy
            pos = self.state_estimator.pos
            vel = self.state_estimator.vel
            vel_i = alpha * target_vel + (1 - alpha) * vel # interpolation of velocity
            vel_i_arr = np.tile(vel_i, (self.MPC_N + 1, 1))
            yaw_i = alpha * target_yaw + (1 - alpha) * rpy[2] # interpolation of yaw angle
            yaw_i_arr = np.tile(yaw_i, (self.MPC_N + 1, 1))
            pos_i = pos + vel_i * self.dt
            pos_i_arr = np.tile(pos_i, (self.MPC_N + 1, 1))

            self.pos_control(pos, rpy, vel, pos_i_arr, vel_i_arr, yaw_i_arr, status=status)

    def hover(self, duration=2.0, target_yaw_deg=0.0, status=Status.HOVER):
        """ Hover the drone at the current position.
        
        Args:
            duration (float): duration of the hover in seconds.
            target_yaw_deg (float): target yaw angle in degrees.
            status (Status): status of the drone.
            
        """
        init_pos = self.state_estimator.pos + np.zeros(3,)
        init_pos_arr = np.tile(init_pos, (self.MPC_N + 1, 1))
        target_vel = np.zeros(3,)
        target_vel_arr = np.tile(target_vel, (self.MPC_N + 1, 1))
        num_steps = int(duration * self.control_freq)
        target_yaw = deg2rad(target_yaw_deg)
        target_yaw_arr = np.tile(target_yaw, (self.MPC_N + 1, 1))

        for i in range(num_steps):
            rpy = self.state_estimator.rpy
            pos = self.state_estimator.pos
            vel = self.state_estimator.vel
            self.pos_control(pos, rpy, vel, init_pos_arr, target_vel_arr, target_yaw_arr, status=status)



   '''higher level behavior'''


    def static_observation(self, hover_duration=5.0):
        """ Keep static to let the variance of observations converge to some small value, especially important in KF
        
        Args: 
            duration (float): duration of the observation in seconds.
            
        """
        self.idle()
        self.hover(hover_duration, target_yaw_deg, status=Status.STATIC_OBSV)

    def take_off(self, velocity, height, target_yaw_deg=0.0, interpolation_duration=1.0, hover_duration=2.0):
        """ Take off the drone to a certain height.
        
        Args: 
            duration (float): duration of the take off in seconds.
            height (float): height to take off to in meters.
            target_yaw_deg (float): target yaw angle in degrees.
            
        """
        # Take off
        # Period 1: acceleration
        target_vel = np.array([0.0, 0.0, velocity])
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.TAKEOFF)
        # Period 2: uniform rectilinear motion
        target_height = height - velocity * interpolation_duration / 2
        self.vertical(velocity, height, target_yaw_deg, status=Status.TAKEOFF)
        # Period 3: deceleration
        target_vel = np.zeros(3,)
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.TAKEOFF)

        # Hover
        self.hover(hover_duration, target_yaw_deg, status=Status.TAKEOFF)

    def land(self, velocity, height=0.03, target_yaw_deg=0.0, interpolation_duration=0.8, hover_duration=2.0):
        """ Land the drone to a certain height.

        Args:
            duration (float): duration of the landing in seconds.
            height (float): height to land to in meters.
            target_yaw_deg (float): target yaw angle in degrees.

        """
        # Firstly go back to the origin in XOY
        # Period 1: acceleration
        pos = self.state_estimator.pos
        vel_x = - velocity * pos[0] / np.linalg.norm([pos[0], pos[1]])
        vel_y = - velocity * pos[1] / np.linalg.norm([pos[0], pos[1]])
        target_vel = np.array([vel_x, vel_y, 0.0])
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.LAND)
        # Period 2: uniform rectilinear motion
        target_x = - vel_x * interpolation_duration / 2
        target_y = - vel_y * interpolation_duration / 2
        self.horizontal(velocity, target_x, target_y, target_yaw_deg, status=Status.LAND)
        # Period 3: deceleration
        target_vel = np.zeros(3,)
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.LAND)
        
        # Hover
        self.hover(hover_duration, target_yaw_deg, status=Status.LAND)

        # Then go back to the origin
        # Period 1: acceleration
        target_vel = np.array([0.0, 0.0, -velocity])
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.LAND)
        # Period 2: uniform rectilinear motion to the origin
        self.vertical(velocity, height, target_yaw_deg, status=Status.LAND)

        self.cf.cmdStop()

    def track_traj(self, traj: TrajectoryGenerator3DPeriodicMotion): 
        """ Track the trajectory generated by the trajectory generator.
        
        Args: traj (TrajectoryGenerator2DPeriodicMotion / TrajectoryGenerator3DPeriodicMotion): trajectory generator object.
    
        """

        # Use linear smoothing between the current velocity and the desired velocity at the beginning of the trajectory
        target_pos, target_vel, target_yaw_rad = traj.get_coordinates(0)
        interpolation_duration = 1.0
        target_yaw_deg = rad2deg(target_yaw_rad)
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.INTERPOLATE)

        rpy = self.state_estimator.rpy
        pos = self.state_estimator.pos
        vel = self.state_estimator.vel

        # Reset the offset of the trajectory to match the current position
        init_target_pos, _, _ = traj.get_coordinates(0)
        offset = pos - init_target_pos
        traj.reset_offset(offset.tolist())

        startTime = self.timeHelper.time()
        time_traj = 0.0

        num_steps = int(traj.traj_length * self.control_freq)

        looping_condition = False
        if self.sim:
            step = 0
            looping_condition = step < num_steps
        else:
            looping_condition = time_traj <= traj.traj_length

        while looping_condition and not rospy.is_shutdown():
            rpy = self.state_estimator.rpy
            pos = self.state_estimator.pos
            vel = self.state_estimator.vel

            if self.sim:
                time_traj = step * self.dt
                step += 1
                looping_condition = step < num_steps
            else:
                time_traj = self.timeHelper.time() - startTime
                looping_condition = time_traj <= traj.traj_length
            

            # Update counter for gradually starting
            self.track_traj_counter += 1


            time_arr = np.arange(time_traj, time_traj + (self.MPC_N + 1) * self.dt, self.dt)
            target_pos_arr = np.zeros((self.MPC_N + 1, 3))
            target_vel_arr = np.zeros((self.MPC_N + 1, 3))
            target_yaw_arr = np.zeros((self.MPC_N + 1, 1))

            for i in range(self.MPC_N + 1):
                if i < self.track_traj_counter:
                    pos_ref, vel_ref, yaw_ref = traj.get_coordinates(time_arr[i])
                    target_pos_arr[i, :] = pos_ref.T
                    target_vel_arr[i, :] = vel_ref.T
                    target_yaw_arr[i, :] = yaw_ref
                else: 
                    # Use gradually starting at initial stage of track_traj
                    target_pos_arr[i, :] = target_pos_arr[i-1, :]
                    target_vel_arr[i, :] = target_vel_arr[i-1, :]
                    target_yaw_arr[i, :] = target_yaw_arr[i-1, :]

            
            self.pos_control(pos, rpy, vel, target_pos_arr, target_vel_arr, target_yaw_arr, status=Status.TRACK_TRAJ)

        # Use linear smoothing between the end of the trajectory and hovering
        target_vel = np.zeros(3,)
        interpolation_duration = 1.0
        target_yaw_deg = 0.0
        self.interpolate_vel(interpolation_duration, target_vel, target_yaw_deg, status=Status.INTERPOLATE)


if __name__ == "__main__":
    import wandb
    import yaml
    import json

    # read config file for crayzflie id
    with open("../launch/crazyflies.yaml", 'r') as stream:
        try:
            cf_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        cf_id_dec = 'cf' + str(cf_config['crazyflies'][0]['id'])

    # Read parameters file for the controller gains
    script_dir = os.path.dirname(__file__)
    param_file = os.path.join(script_dir, "parameters.json")
    with open(param_file) as f:
        params = json.load(f)  

        i_range = params['pos_ctl']['i_range']
        kp = params['pos_ctl']['kp']
        kd = params['pos_ctl']['kd']
        ki = params['pos_ctl']['ki']

    # Initialize Vicon 
    vicon = ViconWatcher(cf_id_dec=cf_id_dec)

    # Initialize State Estimator Subscriber
    state_estimator = StateEstimator()

    # Initialize quadrotor motion
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "Experiments/figure_8/")
    # create file name with date and time stamp
    file_name = "data_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
    file_path = os.path.join(data_dir, file_name)
    print("Data will be saved to: ", file_path)
    control_freq = 60.0
    grad_start = True
    quad_motion = QuadMotion(state_estimator, vicon=vicon, control_freq=control_freq,  grad_start=grad_start, 
                             verbose=True, log_data=True, filename=file_path)

    # Set parameters
    traj_type = "figure8"  # Trajectory type {"circle", "square", "figure8"}
    num_cycles = 2.0  # Number of cycles to complete
    scaling = 0.6  # Trajectory scaling
    traj_length = 10.0  # Trajectory length in seconds
    sample_time = 0.01  # Sampling time, only for plotting
    traj_plane = "xyz"  # Trajectory plane

    wandb.init(project='test', 
               config={'file_path': file_path, 
                       'control_freq': control_freq,
                       'traj_type': traj_type,
                       'num_cycles': num_cycles,
                       'scaling': scaling,
                       'traj_length': traj_length,
                       'sample_time': sample_time,
                       'traj_plane': traj_plane,
                       'i_range': i_range,
                       'kp': kp,
                       'kd': kd,
                       'ki': ki,
                       'is_real': True,})

    # Initialize trajectory generator
    traj = TrajectoryGenerator3DPeriodicMotion(traj_type=traj_type,
                                               num_cycles=num_cycles,
                                               scaling=scaling,
                                               traj_length=traj_length,
                                               sample_time=sample_time,
                                               traj_plane=traj_plane)

    
    velocity = 0.3
    height = 0.7 # 1.5 for Trajectory 4
    target_yaw_deg = 0.0
    observation_duration = 2.0

    # Static observation stage, especially for KF
    quad_motion.static_observation(observation_duration)

    # Take off
    quad_motion.take_off(velocity, height, target_yaw_deg)

    # Track trajectory
    quad_motion.track_traj(traj)

    # Land
    quad_motion.land(velocity)


    # finsih wandb run
    wandb.finish()
