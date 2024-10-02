import numpy as np
import math
import json
import os
from matplotlib import pyplot as plt



class TrajectoryGenerator3DPeriodicMotion():

    def __init__(self, traj_type="figure8", num_cycles=1, scaling=1.0, position_offset=[0.0, 0.0, 0.0], traj_length=10.0, sample_time=0.01, traj_plane="xyz"):
        
        # Read parameters file for the controller gains
        script_dir = os.path.dirname(__file__)
        param_file = os.path.join(script_dir, "parameters.json")
        with open(param_file) as f:
            params = json.load(f)  
        # Define prediction horizon
        N = params['MPC_solver']['N'] # Prediction Horizon

        # Import model parameters
        model_file = '/home/haocheng/Experiments/figure_8/merge_model.json'
        with open(model_file) as file:
            identified_model = json.load(file)
        self.params_acc = identified_model['params_acc']

        # Define Gravitational Acceleration
        self.GRAVITY = 9.806

        # Get trajectory type
        valid_traj_type = ["figure8"]
        if traj_type in valid_traj_type:
            self.traj_type = traj_type
        else:
            raise Exception("Trajectory type should be one of [figure8].")

        # Get number of cycles
        self.num_cycles = num_cycles

        # Get trajectory scaling factor
        self.scaling = scaling

        # Get position offset
        self.position_offset = position_offset

        # Get trajectory length in seconds
        self.traj_length = traj_length # time length for all cycle
        self.traj_period = traj_length / num_cycles # time period for 1 cycle
        self.traj_freq = 2.0 * np.pi / self.traj_period # round frequency for 1 cycle

        # Get sample time
        self.sample_time = sample_time

        # Get trajectory plane
        self.traj_plane = traj_plane
        self.direction_list = ["x", "y", "z"]
        self.set_coordinates_indeces() # call methode to identify axis that has motion

    def reset_offset(self, offset): # may need several reset with in one run, therefore defined seperately
        self.position_offset = offset # data type: list with 3 variable (x, y, z)

    def get_coordinates(self, t): # call calculation methode + offset compensation
        # input: 
        # - t: time for current step in trajectory running, unit: second
        # output: 
        # - pos_ref, vel_ref: reference state, unit: m, m/s
        # - rpy_ref: reference input, unit: rad
        # - thrust: reference input, unit: N
        
        # call function figure8 to calculate the reference state for current step
        coords_a, coords_b, coords_c, coords_a_dot, coords_b_dot, coords_c_dot, roll, pitch, yaw, thrust= self.figure8(t) 

        # Initialize position and velocity references
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        rpy_ref = np.zeros((3,))

        # Set position and velocity references based on the trajectory chosen
        pos_ref[self.coord_index_a] = coords_a + self.position_offset[self.coord_index_a]
        pos_ref[self.coord_index_b] = coords_b + self.position_offset[self.coord_index_b]
        pos_ref[self.coord_index_c] = coords_c + self.position_offset[self.coord_index_c]

        vel_ref[self.coord_index_a] = coords_a_dot
        vel_ref[self.coord_index_b] = coords_b_dot
        vel_ref[self.coord_index_c] = coords_c_dot

        rpy_ref[0] = roll
        rpy_ref[1] = pitch
        rpy_ref[2] = yaw

        return pos_ref, vel_ref, rpy_ref, thrust
    
    def set_coordinates_indeces(self): 
        if self.traj_plane[0] in self.direction_list and self.traj_plane[1] in self.direction_list and self.traj_plane[2] in self.direction_list and len(set(self.traj_plane)) == len(self.traj_plane):
            self.coord_index_a = self.direction_list.index(self.traj_plane[0]) # EG: traj_plane[0] = "Y", coord_index_a = 1
            self.coord_index_b = self.direction_list.index(self.traj_plane[1])
            self.coord_index_c = self.direction_list.index(self.traj_plane[2])
            print("Coords: %s, %s, %s" % (self.coord_index_a, self.coord_index_b, self.coord_index_c))
        else:
            raise Exception("Trajectory plane should be in form of ""ab"", where a and b can be {x, y, z}.")
    
    def get_coordinates_indeces(self):
        return [self.coord_index_a, self.coord_index_b, self.coord_index_c]

    def figure8(self, t): # special curve design

        # Calculate position, velocity and acceleration of each point on trajectory (unit: m, m/s, m/s2)
        '''
        # curve 1
        coords_a = self.scaling * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.cos(self.traj_freq * t)
        coords_c = self.scaling * 0.1 * t

        coords_a_dot = self.scaling * self.traj_freq * np.cos(self.traj_freq * t)
        coords_b_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * 0.1

        coords_a_dot_dot = - self.scaling * self.traj_freq**2 * np.sin(self.traj_freq * t)
        coords_b_dot_dot = - self.scaling * self.traj_freq**2 * np.cos(self.traj_freq * t)
        coords_c_dot_dot = np.zeros_like(t)
        '''

        
        # curve 2
        coords_a = self.scaling * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.cos(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        coords_a_dot = self.scaling * self.traj_freq * np.cos(self.traj_freq * t)
        coords_b_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)

        coords_a_dot_dot = - self.scaling * self.traj_freq**2 * np.sin(self.traj_freq * t)
        coords_b_dot_dot = - self.scaling * self.traj_freq**2 * np.cos(self.traj_freq * t) 
        coords_c_dot_dot = -2 * self.scaling * self.traj_freq**2 * np.sin(2 * self.traj_freq * t) 

        

        '''
        # curve 3
        coords_a = self.scaling * np.sin(self.traj_freq * t) * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.cos(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        coords_a_dot = self.scaling * self.traj_freq * np.sin(2 * self.traj_freq * t)
        coords_b_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)

        coords_a_dot_dot = 2 * self.scaling * self.traj_freq**2 * np.cos(2 * self.traj_freq * t) 
        coords_b_dot_dot = - self.scaling * self.traj_freq**2 * np.cos(self.traj_freq * t) 
        coords_c_dot_dot = -2 * self.scaling * self.traj_freq**2 * np.sin(2 * self.traj_freq * t) 
        '''

        '''
        # curve 4
        coords_a = self.scaling * np.sin(3 * self.traj_freq * t) * np.cos(self.traj_freq * t)
        coords_b = self.scaling * np.sin(3 * self.traj_freq * t) * np.sin(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(3 * self.traj_freq * t)

        coords_a_dot = 3 * self.scaling * self.traj_freq * np.cos(3 * self.traj_freq * t) * np.cos(self.traj_freq * t) - self.scaling * self.traj_freq * np.sin(3 * self.traj_freq * t) * np.sin(self.traj_freq * t)
        coords_b_dot = 3 * self.scaling * self.traj_freq * np.cos(3 * self.traj_freq * t) * np.sin(self.traj_freq * t) + self.scaling * self.traj_freq * np.sin(3 * self.traj_freq * t) * np.cos(self.traj_freq * t)
        coords_c_dot = -3 * self.scaling * self.traj_freq * np.sin(3 * self.traj_freq * t)

        coords_a_dot_dot = -3 * self.scaling * self.traj_freq**2 * (4 * np.sin(3 * self.traj_freq * t) * np.cos(self.traj_freq * t) + 3 * np.cos(3 * self.traj_freq * t) * np.sin(self.traj_freq * t))
        coords_b_dot_dot = 3 * self.scaling * self.traj_freq**2 * (4 * np.sin(3 * self.traj_freq * t) * np.sin(self.traj_freq * t) - 3 * np.cos(3 * self.traj_freq * t) * np.cos(self.traj_freq * t))
        coords_c_dot_dot = -9 * self.scaling * self.traj_freq**2 * np.cos(3 * self.traj_freq * t)  
        '''

        '''
        # curve 5
        k = 0.5
        rho = self.scaling * (1 - k * np.cos(4 * self.traj_freq * t))

        coords_a = rho * np.sin(self.traj_freq * t)
        coords_b = rho * np.cos(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        drhodt = 4 * self.scaling * k * self.traj_freq * np.sin(4 * self.traj_freq * t)
        
        coords_a_dot = drhodt * np.cos(self.traj_freq * t) - rho * self.traj_freq * np.sin(self.traj_freq * t)
        coords_b_dot = drhodt * np.sin(self.traj_freq * t) + rho * self.traj_freq * np.cos(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)

        d2rhodt2 = 16 * self.scaling * k * self.traj_freq**2 * np.cos(4 * self.traj_freq * t)
        coords_a_dot_dot = d2rhodt2 * np.cos(self.traj_freq * t) - 2 * drhodt * self.traj_freq * np.sin(self.traj_freq * t) - rho * self.traj_freq**2 * np.cos(self.traj_freq * t)
        coords_b_dot_dot = d2rhodt2 * np.sin(self.traj_freq * t) + 2 * drhodt * self.traj_freq * np.cos(self.traj_freq * t) - rho * self.traj_freq**2 * np.sin(self.traj_freq * t)
        coords_c_dot_dot = -2 * self.scaling * self.traj_freq**2 * np.sin(2 * self.traj_freq * t)
        '''

        '''
        # curve 6
        coords_a = self.scaling * (np.sin(self.traj_freq * t) ** 3)
        coords_b = self.scaling * (np.cos(self.traj_freq * t) ** 3)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        coords_a_dot = 3 * self.scaling * self.traj_freq * (np.sin(2 * self.traj_freq * t) ** 2) * np.cos(self.traj_freq * t)
        coords_b_dot = - 3 * self.scaling * self.traj_freq * (np.cos(self.traj_freq * t) ** 2) * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)

        coords_a_dot_dot = 3 * self.scaling * self.traj_freq**2 * np.sin(self.traj_freq * t) * (2 - 3 * (np.sin(self.traj_freq * t))**2) 
        coords_b_dot_dot = 3 * self.scaling * self.traj_freq**2 * np.cos(self.traj_freq * t) * (2 - 3 * (np.cos(self.traj_freq * t))**2)
        coords_c_dot_dot = -2 * self.scaling * self.traj_freq**2 * np.sin(2 * self.traj_freq * t)
        '''

        
        '------input reference------'

        # Methode 1: Calculate accurate input reference
        #         +: will lead to no-deviation tracking
        #         -: need analytic expression of trajectory known adn the reference need to be 2-order differentiable
        # Assumption: 1) reference trajectory is 2-order differentiable; 
        #             2) analytic expression of trajectory known; 
        #             3) yaw == 0;
        # Part 1: Calculate pose of trajectory (unit: rad)
        #roll = 0 #np.arcsin(- coords_b_dot_dot / np.linalg.norm([coords_a_dot_dot, coords_b_dot_dot, coords_c_dot_dot + self.GRAVITY]))
        #pitch = 0 #np.arcsin(coords_a_dot_dot / (np.linalg.norm([coords_a_dot_dot, coords_b_dot_dot, coords_c_dot_dot + self.GRAVITY]) * np.cos(roll)))
        #yaw = 0
        # Part 2: Calculate command of thrust (unit: N)
        #thrust = (np.linalg.norm([coords_a_dot_dot, coords_b_dot_dot, coords_c_dot_dot + self.GRAVITY]) - self.params_acc[1]) / self.params_acc[0]

        # Method 2: Set a rough estimation as input reference
        #         +: no additional request on reference trajectory
        #         -: lower tracking accurancy
        roll = 0 
        pitch = 0 
        yaw = 0
        thrust = (self.GRAVITY - self.params_acc[1]) / self.params_acc[0] # use hover thrust as estimation: thrust_collective = m * g
        
        
        return coords_a, coords_b, coords_c, coords_a_dot, coords_b_dot, coords_c_dot, roll, pitch, yaw, thrust


    def preview_trajectory(self): # tbd
        pass
