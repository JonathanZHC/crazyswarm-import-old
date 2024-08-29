import numpy as np
import math
import json
import os
from matplotlib import pyplot as plt


class TrajectoryGenerator2DPeriodicMotion():

    def __init__(self, traj_type="figure8", num_cycles=1, scaling=1.0, position_offset=[0.0, 0.0, 0.0], traj_length=10.0, sample_time=0.01, traj_plane="xy"):
        # Get trajectory type
        valid_traj_type = ["circle", "square", "figure8"] # figure 8 for any other trajectory than circle / square
        if traj_type in valid_traj_type:
            self.traj_type = traj_type
        else:
            raise Exception("Trajectory type should be one of [circle, square, figure8].")

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
        # t: time for current step in trajectory running, unit: second
        # Get coordinates
        if self.traj_type == "figure8": # call function figure8 to calculate the reference state for current step
            coords_a, coords_b, coords_a_dot, coords_b_dot = self.figure8(t)
        elif self.traj_type == "circle": # call function circle to calculate the reference state for current step
            coords_a, coords_b, coords_a_dot, coords_b_dot = self.circle(t)
        elif self.traj_type == "square": # call function square to calculate the reference state for current step
            coords_a, coords_b, coords_a_dot, coords_b_dot = self.square(t)

        # Initialize position and velocity references
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        roll_ref = np.zeros((1,))

        # Set position and velocity references based on the trajectory chosen
        pos_ref[self.coord_index_a] = coords_a + self.position_offset[self.coord_index_a]
        pos_ref[self.coord_index_b] = coords_b + self.position_offset[self.coord_index_b]
        pos_ref[self.coord_index_c] = self.position_offset[self.coord_index_c]

        vel_ref[self.coord_index_a] = coords_a_dot
        vel_ref[self.coord_index_b] = coords_b_dot

        return pos_ref, vel_ref, roll_ref

    def set_coordinates_indeces(self): 
        if self.traj_plane[0] in self.direction_list and self.traj_plane[1] in self.direction_list and self.traj_plane[0] != self.traj_plane[1]:
            self.coord_index_a = self.direction_list.index(self.traj_plane[0]) # EG: traj_plane[0] = "Y", coord_index_a = 1
            self.coord_index_b = self.direction_list.index(self.traj_plane[1])
            # Get the remaining direction
            self.coord_index_c = self.direction_list.index(list(set(self.direction_list) - set(list(self.traj_plane)))[0])
            print("Coords: %s, %s, %s" % (self.coord_index_a, self.coord_index_b, self.coord_index_c))
        else:
            raise Exception("Trajectory plane should be in form of ""ab"", where a and b can be {x, y, z}.")
        
    def get_coordinates_indeces(self):
        return [self.coord_index_a, self.coord_index_b, self.coord_index_c]

    def figure8(self, t): # special curve design
        coords_a = self.scaling * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.sin(self.traj_freq * t) * np.cos(self.traj_freq * t)

        coords_a_dot = self.scaling * self.traj_freq * np.cos(self.traj_freq * t)
        coords_b_dot = self.scaling * self.traj_freq * (np.cos(self.traj_freq * t) ** 2 - np.sin(self.traj_freq * t) ** 2)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def circle(self, t):
        coords_a = self.scaling * np.cos(self.traj_freq * t)
        coords_b = self.scaling * np.sin(self.traj_freq * t)

        coords_a_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_b_dot = self.scaling * self.traj_freq * np.cos(self.traj_freq * t)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def square(self, t):
        # Compute time for the cycle
        cycle_time = t % self.traj_period

        # Compute time for each segment to complete
        segment_period = self.traj_period / 4.0

        # Check current segment index
        segment_index = int(np.floor(cycle_time / segment_period))

        # Check time along the current segment and ratio of completion
        segment_time = cycle_time % segment_period
        traverse_speed = self.scaling / segment_period

        # Position along segment
        segment_position = traverse_speed * segment_time

        if segment_index == 0:
            # Moving up along second axis from (0, 0)
            coords_a = 0.0
            coords_b = segment_position
            coords_a_dot = 0.0
            coords_b_dot = traverse_speed
        elif segment_index == 1:
            # Moving left along first axis from (0, 1)
            coords_a = -segment_position
            coords_b = self.scaling
            coords_a_dot = -traverse_speed
            coords_b_dot = 0.0
        elif segment_index == 2:
            # Moving down along second axis from (-1, 1)
            coords_a = -self.scaling
            coords_b = self.scaling - segment_position
            coords_a_dot = 0.0
            coords_b_dot = -traverse_speed
        elif segment_index == 3:
            # Moving right along second axis from (-1, 0)
            coords_a = -self.scaling + segment_position
            coords_b = 0.0
            coords_a_dot = traverse_speed
            coords_b_dot = 0.0

        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def preview_trajectory(self):
        # Get time stamps
        times = np.arange(0, self.traj_length, self.sample_time)

        # Get time-discretized trajectory
        pos_ref_traj, vel_ref_traj = self.get_coordinates(times[0])
        speed_traj = np.linalg.norm(vel_ref_traj)
        for k in range(1, len(times)):
            pos_ref_temp, vel_ref_temp = self.get_coordinates(times[k])
            pos_ref_traj = np.hstack((pos_ref_traj, pos_ref_temp))
            vel_ref_traj = np.hstack((vel_ref_traj, vel_ref_temp))
            speed_traj = np.hstack((speed_traj, np.linalg.norm(vel_ref_temp)))

        # Print basic properties
        print("Trajectory type: %s" % self.traj_type)
        print("Trajectory plane: %s" % self.traj_plane)
        print("Trajectory length: %s sec" % self.traj_length)
        print("Number of cycles: %d" % self.num_cycles)
        print("Position bounds: x [%.2f, %.2f] m, y [%.2f, %.2f] m, z [%.2f, %.2f] m"
              % (min(pos_ref_traj[0]), max(pos_ref_traj[0]),
                 min(pos_ref_traj[1]), max(pos_ref_traj[1]),
                 min(pos_ref_traj[2]), max(pos_ref_traj[2])))

        print("Velocity bounds: vx [%.2f, %.2f] m/s, vy [%.2f, %.2f] m/s, vz [%.2f, %.2f] m/s"
              % (min(vel_ref_traj[0]), max(vel_ref_traj[0]),
                 min(vel_ref_traj[1]), max(vel_ref_traj[1]),
                 min(vel_ref_traj[2]), max(vel_ref_traj[2])))

        print("Speed: min %.2f m/s max %.2f m/s mean %.2f" % (min(speed_traj), max(speed_traj), np.mean(speed_traj)))

        print("Trajectory period: %.2f sec" % self.traj_period)
        print("Angular speed: %.2f rad/sec" % self.traj_freq)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(pos_ref_traj[0], pos_ref_traj[1], pos_ref_traj[2])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.show()

        return pos_ref_traj, vel_ref_traj



class TrajectoryGenerator3DPeriodicMotion():

    def __init__(self, traj_type="figure8", num_cycles=1, scaling=1.0, position_offset=[0.0, 0.0, 0.0], traj_length=10.0, sample_time=0.01, traj_plane="xyz"):
        
        # Read parameters file for the controller gains
        script_dir = os.path.dirname(__file__)
        param_file = os.path.join(script_dir, "parameters.json")
        with open(param_file) as f:
            params = json.load(f)  
        # Define prediction horizon
        N = params['MPC_solver']['N'] # Prediction Horizon

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

    def get_coordinates(self, t): # Get coordinates: call calculation methode + offset compensation
        # t: time for current step in trajectory running, unit: second
        
        # call function figure8 to calculate the reference state for current step
        coords_a, coords_b, coords_c, coords_a_dot, coords_b_dot, coords_c_dot, roll= self.figure8(t) 

        # Initialize position and velocity references
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        roll_ref = np.zeros((1,))

        # Set position and velocity references based on the trajectory chosen
        pos_ref[self.coord_index_a] = coords_a + self.position_offset[self.coord_index_a]
        pos_ref[self.coord_index_b] = coords_b + self.position_offset[self.coord_index_b]
        pos_ref[self.coord_index_c] = coords_c + self.position_offset[self.coord_index_c]

        vel_ref[self.coord_index_a] = coords_a_dot
        vel_ref[self.coord_index_b] = coords_b_dot
        vel_ref[self.coord_index_c] = coords_c_dot

        roll_ref = roll

        return pos_ref, vel_ref, roll_ref
    
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
        
        '''
        # curve 1
        coords_a = self.scaling * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.cos(self.traj_freq * t)
        coords_c = self.scaling * 0.1 * t

        coords_a_dot = self.scaling * self.traj_freq * np.cos(self.traj_freq * t)
        coords_b_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * 0.1
        '''

        '''
        # curve 2
        coords_a = self.scaling * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.cos(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        coords_a_dot = self.scaling * self.traj_freq * np.cos(self.traj_freq * t)
        coords_b_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)
        '''

        
        # curve 3
        coords_a = self.scaling * np.sin(self.traj_freq * t) * np.sin(self.traj_freq * t)
        coords_b = self.scaling * np.cos(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        coords_a_dot = self.scaling * self.traj_freq * np.sin(2 * self.traj_freq * t)
        coords_b_dot = - self.scaling * self.traj_freq * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)
        

        '''
        # curve 4
        coords_a = self.scaling * np.sin(3 * self.traj_freq * t) * np.cos(self.traj_freq * t)
        coords_b = self.scaling * np.sin(3 * self.traj_freq * t) * np.sin(self.traj_freq * t)
        coords_c = self.scaling / 2 + self.scaling * np.cos(3 * self.traj_freq * t)

        coords_a_dot = 3 * self.scaling * self.traj_freq * np.cos(3 * self.traj_freq * t) * np.cos(self.traj_freq * t) - self.scaling * self.traj_freq * np.sin(3 * self.traj_freq * t) * np.sin(self.traj_freq * t)
        coords_b_dot = 3 * self.scaling * self.traj_freq * np.cos(3 * self.traj_freq * t) * np.sin(self.traj_freq * t) + self.scaling * self.traj_freq * np.sin(3 * self.traj_freq * t) * np.cos(self.traj_freq * t)
        coords_c_dot = -3 * self.scaling * self.traj_freq * np.sin(3 * self.traj_freq * t)
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
        '''

        '''
        # curve 6
        coords_a = self.scaling * (np.sin(self.traj_freq * t) ** 3)
        coords_b = self.scaling * (np.cos(self.traj_freq * t) ** 3)
        coords_c = self.scaling / 2 + self.scaling * np.cos(self.traj_freq * t) * np.sin(self.traj_freq * t)

        coords_a_dot = 3 * self.scaling * self.traj_freq * (np.sin(2 * self.traj_freq * t) ** 2) * np.cos(self.traj_freq * t)
        coords_b_dot = - 3 * self.scaling * self.traj_freq * (np.cos(self.traj_freq * t) ** 2) * np.sin(self.traj_freq * t)
        coords_c_dot = self.scaling * self.traj_freq * np.cos(2 * self.traj_freq * t)
        '''
        
        roll = 0

        return coords_a, coords_b, coords_c, coords_a_dot, coords_b_dot, coords_c_dot, roll

    def preview_trajectory(self): # tbd
        pass
