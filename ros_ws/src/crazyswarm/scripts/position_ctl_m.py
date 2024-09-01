import numpy as np
import math

from helper import rotation2euler, euler2rotation, clamp, pwm2thrust, thrust2pwm

class PositionController:
    def __init__(self, quad=None):
        if not quad is None:
            # get parameters
            self.params = quad.params
            self.controller_params = quad.controller

        self.i_error = 0

    def compute_action(self, measured_pos, measured_rpy, measured_vel, desired_pos, desired_yaw, desired_vel, dt):
        """Compute the thrust and euler angles for the drone to reach the desired position.
        
        Args:
            measured_pos (np.array): current position of the drone
            measured_rpy (np.array): current roll, pitch, yaw angles of the drone
            desired_pos (np.array): desired position of the drone
            desired_yaw (float): desired yaw angle of the drone in radians
            dt (float): time step
            
        Returns:
            thrust_desired (float): desired thrust
            euler_desired (np.array): desired euler angles
        """
        current_R = euler2rotation(measured_rpy[0], measured_rpy[1], measured_rpy[2])

        # compute position and velocity error
        pos_error = desired_pos - measured_pos
        vel_error = desired_vel - measured_vel

        # update integral error
        self.i_error += pos_error * dt
        self.i_error = clamp(self.i_error, np.array(self.params.pos_ctl.i_range))

        # compute target thrust (PID Controller)
        target_thrust = np.zeros(3)

        target_thrust += self.params.pos_ctl.kp * pos_error
        target_thrust += self.params.pos_ctl.ki * self.i_error
        target_thrust += self.params.pos_ctl.kd * vel_error
        # target_thrust += params.quad.m * desired_acc
        target_thrust[2] += self.params.quad.m * self.params.quad.g # 16
        
        # update z_axis#
        z_axis = current_R[:,2]

        # update current thrust
        current_thrust = target_thrust.dot(z_axis) # 17
        current_thrust = max(current_thrust, 0.3 * self.params.quad.m * self.params.quad.g) # incase too large deceleration
        current_thrust = min(current_thrust, 1.8 * self.params.quad.m * self.params.quad.g) # incase too large acceleration
        # print('current_thrust:', current_thrust)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust) # 19
        x_c_des = np.array([math.cos(desired_yaw), math.sin(desired_yaw), 0.0]) # 20
        y_axis_desired = np.cross(z_axis_desired, x_c_des) # 20
        y_axis_desired /= np.linalg.norm(y_axis_desired) # 20
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired) # 21

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T # 17
        euler_desired = rotation2euler(R_desired)

        pwm_desired = thrust2pwm(current_thrust)

        return pwm_desired, euler_desired

    def position_controller_reset(self):
        self.i_error = np.zeros(3)