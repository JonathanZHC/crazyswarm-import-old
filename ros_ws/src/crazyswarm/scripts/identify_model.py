import json
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.linalg import expm
from scipy.signal import butter, filtfilt

from helper import pwm2thrust, thrust2pwm

from position_ctl_m import PositionController
from utils import DataVarIndex, Status, load_data, get_file_path_from_run, var_bounds, GRAVITY
from plot_data import Plotter


class DataSmoother:
    """A class that smooths the recorded data."""
    
    def __init__(self, file_path, status=None):
        # Create a dictionary to match the values with their derivatives, e.g. POS_X -> VEL_X
        self.match_first_derivative = {
            DataVarIndex.POS_X: DataVarIndex.VEL_X,
            DataVarIndex.POS_Y: DataVarIndex.VEL_Y,
            DataVarIndex.POS_Z: DataVarIndex.VEL_Z,
            DataVarIndex.ROLL: DataVarIndex._RATE,
            DataVarIndex.PITCH: DataVarIndex.PITCH_RATE,
            DataVarIndex.YAW: DataVarIndex.YAW_RATE,
        }
        self.match_second_derivative = {    
            DataVarIndex.POS_X: DataVarIndex.ACC_X,
            DataVarIndex.POS_Y: DataVarIndex.ACC_Y,
            DataVarIndex.POS_Z: DataVarIndex.ACC_Z,
            DataVarIndex.ROLL: DataVarIndex.ROLL_ACC,
            DataVarIndex.PITCH: DataVarIndex.PITCH_ACC,
            DataVarIndex.YAW: DataVarIndex.YAW_ACC,
        }
        self.match_cmd = {    
            DataVarIndex.ROLL: DataVarIndex.CMD_ROLL,
            DataVarIndex.PITCH: DataVarIndex.CMD_PITCH,
            DataVarIndex.YAW: DataVarIndex.CMD_YAW,
        }

        self.derivatives = {1: self.match_first_derivative, 2: self.match_second_derivative}

        # Load the data from the csv file
        self.data = load_data(file_path)
        if status is not None:
            # Only use the data that matches the status
            self.data = self.data[self.data[:, DataVarIndex.STATUS] == status.value]
 
        self.smoothed = False # mark state

        self.time = self.data[:, DataVarIndex.TIME] # array-like
        self.dt = np.mean(np.diff(self.time)) # numpy.diff -> calculate every time difference along the variable "time"
                                              # only useful within real-run, for simulation 'dt' is always a constant
        num_samples = self.time.shape[0] # function 'shape': dimensions of a array, shape[i] stand for the number of elements in dimension i
        self.time_interpolated = np.arange(num_samples) * self.dt + self.time[0] # theoritical time serious (not the real one), used for interpolation
        print("Sampling time: ", self.dt)
        print("Num samples: ", num_samples)

        self.splines = {}

        self.pos_control = PositionController()

    def smooth_data(self, indices):
        """Smooth the data using cubic spline interpolation."""
        
        # Smooth the data
        for index in indices: # indices stand for the name of data set, like pos_x, vel_x, cmd_thrust, etc.
            # Get the data that needs to be smoothed
            data = self.data[:, index]

            # Convert the thrust from pwm to force
            if index == DataVarIndex.CMD_THRUST:
                data = pwm2thrust(data)

            # use low-pass filter 
            N = 4
            Wn = 0.1
            b, a = butter(N, Wn, 'low')
            # apply butterfilter
            data_smoothed = filtfilt(b, a, data)
            self.data[:, index] = data_smoothed

            # Create a cubic spline interpolation function
            # # Methode1: function 'CubicSpline'
            #data_spline = interpolate.CubicSpline(self.time, data_smoothed)
            #data_interp = data_spline(self.time_interpolated)
            #self.splines[index] = data_spline

            # # Methode 2: function 'splrep'
            data_spline = interpolate.splrep(self.time, data_smoothed, k=5, s=0.001) 
            # Parameters:
            #   k: highest degree for spline fitting, here k = 3 stand for cubic
            #   s: how smooth the curve is, default: s = 0.0
            # Return value: 
            #   tuple (t, c, k) -> representation of analytical description of interpolated curve
            data_interp = interpolate.BSpline(*data_spline)(self.time_interpolated)
            # Parameters:
            #   '* data_spline' stand for tuple (t, c, k)
            #   time points on which interpolated values are going to be solved
            # Return value: 
            #   interpolated values on time points 'time_inetrpolated'

            # store spline
            self.splines[index] = data_spline

            # Update the data
            self.data[:, index] = data_interp
        

        self.smoothed = True # update mark state
    
        # Smooth the derivatives using the splines
        self.smooth_derivatives()

    def smooth_derivatives(self):
        """Smooth the derivatives by taking the derivative of the spline interpolation."""
        if not self.smoothed:
            raise ValueError("Data has to be smoothed first.")
        
        for der in self.derivatives.keys(): # line 35, der = 1 / 2
            match_derivative = self.derivatives[der]
            for index in match_derivative.keys(): # line 18 - 33
                derivative_index = match_derivative[index]

                # Get the spline interpolation function
                spline = self.splines[index]
                # Take the derivative of the spline interpolation
                data_derivative_interp = interpolate.splev(self.time_interpolated, spline, der=der)
                # Parameters:
                #   time points on which interpolated values are going to be solved
                #   '* data_spline' stand for tuple (t, c, k)        
                #   der: order of derivative, here are 1 or 2        
                # Return value: 
                #   set-order (1 / 2) derivative values of interpolated curve on time points 'time_inetrpolated'

                # use low-pass filter 
                N = 4
                Wn = 0.1
                b, a = butter(N, Wn, 'low')
                # apply butterfilter
                data_derivative_interp_smoothed = filtfilt(b, a, data_derivative_interp)

                # Update the data
                self.data[:, derivative_index] = data_derivative_interp_smoothed

    def save_data(self, file_path):
        """Save the smoothed data to a csv file."""
        if not self.smoothed:
            raise ValueError("Data has to be smoothed first.")
        
        # Save the data to a csv file with DataVarIndex as the header
        pd_data = pd.DataFrame(self.data, columns=[var.name for var in DataVarIndex])

        # Replace the status with the actual status string
        pd_data[DataVarIndex.STATUS.name] = pd_data[DataVarIndex.STATUS.name].apply(lambda s: Status(s).name)
        pd_data.to_csv(file_path, index=False)
        
        print("Data smoothed and saved to: ", file_path)

        return file_path


class ModelIdentifier:
    """A class that identifies the model parameters from the smoothed data."""
    
    def __init__(self, file_path, indices, status=None, batch=False, file_path_merged=None):
        self.smooth_indices = indices

        if batch == False: # file_path: list with one node of path, 
            # Smooth the data
            self.smoother = DataSmoother(file_path, status)
            self.smoother.smooth_data(self.smooth_indices) # regulate time points, calculate data of smoothed curve and their 1 / 2 - derivatives

            # Save the smoothed data
            file_path_smoothed = file_path.replace(".csv", "_smoothed.csv") # define name of new file
            self.smooth_data_path_ = self.smoother.save_data(file_path_smoothed)
        
        elif batch == True:

            for file_path_cur in file_path: # file_path: list with several nodes of path, file_path_merged: new file name to store merged data
                if file_path_cur == file_path[0]: 
                    # create object 'smoother' to smooth and store the data
                    self.smoother = DataSmoother(file_path_cur, status)
                    self.smoother.smooth_data(self.smooth_indices)
                else:
                    # create object 'smoother_cur' to smooth and store the data
                    self.smoother_cur = DataSmoother(file_path_cur, status)
                    self.smoother_cur.smooth_data(self.smooth_indices)
                    
                    # delete the table titles in data list
                    row_index = 0
                    np.delete(self.smoother_cur.data, row_index, 0)

                    # merge data in 'smoother_cur' to 'smoother'
                    self.smoother.data = np.vstack((self.smoother.data, self.smoother_cur.data))

            # Save the smoothed data
            self.smooth_data_path_ = self.smoother.save_data(file_path_merged)

        self.params = {}
    
    @property
    def smooth_data_path(self): # get_methode, read-only
        return self.smooth_data_path_

    def identify_model_3D(self, input_indices, output_indices, status=None):
        """Identify the model parameters from the smoothed data."""
        # initialize I/O of system dynamic, will be used in check_model
        self.input_indices = input_indices
        self.output_indices = output_indices

        if status is not None: 
            # Only plot the data that matches the status
            self.smoother.data = self.smoother.data[self.smoother.data[:, DataVarIndex.STATUS] == status.value]
            print("Used samples: ", self.smoother.data.shape[0])


        # # First, identify the model parameters that map thrust to acceleration in x and z
        # Assume a model of the form:
        # normed_acc = params[0] * normed_thrust + params[1] = [normed_thrust(k), 1] * [params[0]; params[1]]
        # where normed_acc = sqrt(acc_x^2 + acc_y^2 + (acc_z + g)^2) and normed_thrust = pwm2thrust(thrust_cmd)

        # Get the input data for acceleration in x, y and z
        thrust_data = self.smoother.data[:, DataVarIndex.CMD_THRUST]
        normed_thrust = np.expand_dims(thrust_data, axis=1) # to represent 'normed_thrust(k) = [normed_thrust(k), 1]'
        # Add a column of ones to the input data
        normed_thrust = np.hstack((normed_thrust, np.ones((normed_thrust.shape[0], 1))))
        
        # Get the acceleration data in x, y and z
        acc_indices = [DataVarIndex.ACC_X, DataVarIndex.ACC_Y, DataVarIndex.ACC_Z]
        acc_data = self.smoother.data[:, acc_indices]
        acc_data[:, 2] += GRAVITY
        normed_acc = np.linalg.norm(acc_data, axis=1) # solve by row
        # normed_acc = np.expand_dims(normed_acc, axis=1) 

        # Identify the linear model parameters using least squares
        self.params["acc"] = np.linalg.lstsq(normed_thrust, normed_acc, rcond=None)[0]

        
        # # Second, identify the model parameters that map the pitch and commanded pitch to the pitch rate
        # # Methode 1: Assume a model of the form:
        #   pitch_rate = params[0] * pitch + params[1] * cmd_pitch
        #   where pitch_rate = d(pitch)/dt
        
        # # Get the input data for pitch and commanded pitch
        pitch_data = self.smoother.data[:, DataVarIndex.PITCH]
        cmd_pitch_data = self.smoother.data[:, DataVarIndex.CMD_PITCH] 
        pitch_data = np.expand_dims(pitch_data, axis=1)
        cmd_pitch_data = np.expand_dims(cmd_pitch_data, axis=1)
        pitch_inputs = np.hstack((pitch_data, cmd_pitch_data))
        # # Get the pitch rate data
        pitch_rate_data = self.smoother.data[:, DataVarIndex.PITCH_RATE]
        pitch_rate_data = np.expand_dims(pitch_rate_data, axis=1)
        # # Identify the linear model parameters using least squares
        self.params["pitch_rate"] = np.linalg.lstsq(pitch_inputs, pitch_rate_data, rcond=None)[0]

        # # Get the input data for roll and commanded roll
        roll_data = self.smoother.data[:, DataVarIndex.ROLL]
        cmd_roll_data = self.smoother.data[:, DataVarIndex.CMD_ROLL]
        roll_data = np.expand_dims(roll_data, axis=1)
        cmd_roll_data = np.expand_dims(cmd_roll_data, axis=1)
        roll_inputs = np.hstack((roll_data, cmd_roll_data))
        # # Get the roll rate data
        roll_rate_data = self.smoother.data[:, DataVarIndex.ROLL_RATE]
        roll_rate_data = np.expand_dims(roll_rate_data, axis=1)
        # # Identify the linear model parameters using least squares
        self.params["roll_rate"] = np.linalg.lstsq(roll_inputs, roll_rate_data, rcond=None)[0]

        # # Get the input data for yaw and commanded yaw
        yaw_data = self.smoother.data[:, DataVarIndex.YAW]
        cmd_yaw_data = self.smoother.data[:, DataVarIndex.CMD_YAW]
        yaw_data = np.expand_dims(yaw_data, axis=1)
        cmd_yaw_data = np.expand_dims(cmd_yaw_data, axis=1)
        yaw_inputs = np.hstack((yaw_data, cmd_yaw_data))
        # # Get the yaw rate data
        yaw_rate_data = self.smoother.data[:, DataVarIndex.YAW_RATE]
        yaw_rate_data = np.expand_dims(yaw_rate_data, axis=1)
        # # Identify the linear model parameters using least squares
        self.params["yaw_rate"] = np.linalg.lstsq(yaw_inputs, yaw_rate_data, rcond=None)[0]
        

        '''
        # # Methode 3: The following apprxoimation works better than the identified model
        # The approximation assumes that the commanded pitch is reached at the next time step
        # Assume a model of the form:
        #   roll_rate = (roll_next - roll) / dt = (cmd_roll - roll) / dt
        #   yaw_rate = (yaw_next - yaw) / dt = (cmd_yaw - yaw) / dt
        #   pitch_rate = (pitch_next - pitch) / dt = (cmd_pitch - pitch) / dt = params[0] * pitch + params[1] * cmd_pitch
        self.params["pitch_rate"] = np.array([- 1.0 / self.smoother.dt, 1.0 / self.smoother.dt])
        self.params["roll_rate"] = self.params["pitch_rate"]
        self.params["yaw_rate"] = self.params["pitch_rate"]
        '''

        # Save the identified model parameters
        identified_model_path = self.smooth_data_path.replace("_smoothed.csv", "_model.json")
        self.save_model(identified_model_path)

    def save_model(self, file_path):
        """Save the identified model parameters to a json file."""
        # Save the model parameters, data indices and the used data file names
        model_data = {
            "params_pitch_rate": self.params["pitch_rate"].tolist(),
            "params_roll_rate": self.params["roll_rate"].tolist(),
            "params_yaw_rate": self.params["yaw_rate"].tolist(),
            "params_acc": self.params["acc"].tolist(),
            "input_indices": [var.name for var in self.input_indices],
            "output_indices": [var.name for var in self.output_indices],
            "data_file": self.smooth_data_path,
            "smooth_indices": [var.name for var in self.smooth_indices],
        }

        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=4)

        print("Model parameters saved to: ", file_path)


if __name__ == "__main__":
    wandb_project = "test"

    # Get the file path from the run
    # Specify the data by setting either the run_name or the file_name
    file_name = None
    run_name = 'dashing-water-47' # easy-star-3 or major-valley-78
    use_latest = False
    smoothed = False

    batch = True # use batch identification or not: False or True

    # group 1
    run_name_list = ['whole-cherry-23', 'colorful-sun-58', 'lemon-grass-24', 'apricot-darkness-33', 'fluent-night-59', 'gentle-dew-60'
                   , 'cool-music-34', 'treasured-waterfall-61', 'vocal-mountain-62', 'wandering-bush-63', 'earthy-water-21', 'glad-haze-65'
                   , 'legendary-aardvark-66', 'balmy-resonance-67', 'frosty-elevator-69', 'kind-firebrand-56', 'sandy-blaze-46', 'sunny-bird-22'
                   , 'rosy-star-72', 'flowing-spaceship-73', 'quiet-leaf-75', 'young-breeze-36', 'stoic-dragon-57', 'vague-snowball-27']

    file_path_merged_smoothed = '/home/haocheng/Experiments/figure_8/merge_smoothed.csv' # define name of new file of merged data
    
    
    # dictionary that maps the trajectory plane to the corresponding indices
    plane2indices_pos = {'x': DataVarIndex.POS_X, 'y': DataVarIndex.POS_Y, 'z': DataVarIndex.POS_Z}
    plane2indices_vel = {'x': DataVarIndex.VEL_X, 'y': DataVarIndex.VEL_Y, 'z': DataVarIndex.VEL_Z}
    plane2indices_acc = {'x': DataVarIndex.ACC_X, 'y': DataVarIndex.ACC_Y, 'z': DataVarIndex.ACC_Z}

    # Select which indices to smooth
    smooth_indices = [
        DataVarIndex.POS_X, 
        DataVarIndex.POS_Y, 
        DataVarIndex.POS_Z,
        DataVarIndex.ROLL,
        DataVarIndex.PITCH,
        DataVarIndex.YAW,
        DataVarIndex.VEL_X,
        DataVarIndex.VEL_Y,
        DataVarIndex.VEL_Z,
        DataVarIndex.ROLL_RATE,
        DataVarIndex.PITCH_RATE,
        DataVarIndex.YAW_RATE,
        DataVarIndex.CMD_THRUST,
        DataVarIndex.CMD_ROLL,
        DataVarIndex.CMD_PITCH,
        DataVarIndex.CMD_YAW
    ]

    # Use one data set from given run_name or merge data from different data set to do batch-identification
    if batch == False: # Identification on one data set: directly use file_path
        file_path, traj_plane = get_file_path_from_run(wandb_project, run_name, file_name, use_latest, smoothed)
        # Initialize the object of class ModelIdentifier
        model_identifier = ModelIdentifier(file_path, smooth_indices, status=None, batch=batch) 
        
    elif batch == True: # Batch-identification: get all file_path -> data merge -> create new file path 
        file_path_batch = [] # initialize the list of all file_path
        for run_name_cur in run_name_list:

            file_path_cur, traj_plane = get_file_path_from_run(wandb_project, run_name_cur, file_name, use_latest, smoothed)
            file_path_batch.append(file_path_cur) # append the current file_path to the file_path list
        # Initialize the object of class ModelIdentifier
        model_identifier = ModelIdentifier(file_path_batch, smooth_indices, status=None, batch=batch, file_path_merged=file_path_merged_smoothed)

    # Select the indices based on the trajectory plane
    data_index_a = plane2indices_pos[traj_plane[0]]
    data_index_b = plane2indices_pos[traj_plane[1]]
    data_index_c = plane2indices_pos[traj_plane[2]]
    data_index_a_vel = plane2indices_vel[traj_plane[0]]
    data_index_b_vel = plane2indices_vel[traj_plane[1]]
    data_index_c_vel = plane2indices_vel[traj_plane[2]]
    data_index_a_acc = plane2indices_acc[traj_plane[0]]
    data_index_b_acc = plane2indices_acc[traj_plane[1]]
    data_index_c_acc = plane2indices_acc[traj_plane[2]]

    plot_indices = [#(data_index_a, data_index_b), 
                    #(data_index_b, data_index_c), 
                    #(data_index_a, data_index_c), 
                    #data_index_a, 
                    #data_index_b,
                    #data_index_c,
                    #data_index_a_vel,
                    #data_index_b_vel,
                    #data_index_c_vel,
                    DataVarIndex.ROLL,
                    DataVarIndex.YAW,
                    DataVarIndex.PITCH,                    
                    DataVarIndex.CMD_THRUST,
                    DataVarIndex.ROLL_RATE,
                    DataVarIndex.YAW_RATE,
                    DataVarIndex.PITCH_RATE,
                    #data_index_a_acc,
                    #data_index_b_acc,
                    #data_index_c_acc,
                    ]

    input_indices = [DataVarIndex.CMD_THRUST, DataVarIndex.CMD_ROLL, DataVarIndex.CMD_YAW, DataVarIndex.CMD_PITCH]
    # The order of the output indices is important: the first output index should be the pitch rate, because the other indices use the pitch
    output_indices = [DataVarIndex.PITCH_RATE, DataVarIndex.ROLL_RATE, DataVarIndex.YAW_RATE, DataVarIndex.ACC_X, DataVarIndex.ACC_Y, DataVarIndex.ACC_Z]
    model_identifier.identify_model_3D(input_indices, output_indices, status=Status.TRACK_TRAJ)


    # plot all data: raw -> smoothed -> simulation
    if batch == False:
        plotter = Plotter(save_fig=False)
        # Plot the raw data
        print("Plotting data from: ", file_path)
        plotter.plot_data(file_path, plot_indices=plot_indices, status=Status.TRACK_TRAJ)

        # Plot the smoothed data
        plotter = Plotter(save_fig=False)
        file_path_smoothed = model_identifier.smooth_data_path
        print("Plotting data from: ", file_path_smoothed)
        plotter.plot_data(file_path_smoothed, plot_indices, status=Status.TRACK_TRAJ)
    
