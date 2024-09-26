import os
import numpy as np
import matplotlib.pyplot as plt


from utils import DataVarIndex, Status, load_data, get_file_path_from_run


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


class Plotter:
    """A class that plots the recorded data."""

    def __init__(self, save_fig=False):
        self.save_fig = save_fig # save generated figure or not

        # Create a dictionary to match the actual value with the desired value
        self.match_desired = {
            DataVarIndex.POS_X: DataVarIndex.DES_POS_X,
            DataVarIndex.POS_Y: DataVarIndex.DES_POS_Y,
            DataVarIndex.POS_Z: DataVarIndex.DES_POS_Z,
            DataVarIndex.ROLL: DataVarIndex.CMD_ROLL,
            DataVarIndex.PITCH: DataVarIndex.CMD_PITCH,
            DataVarIndex.YAW: DataVarIndex.CMD_YAW,
            DataVarIndex.VEL_X: DataVarIndex.DES_VEL_X,
            DataVarIndex.VEL_Y: DataVarIndex.DES_VEL_Y,
            DataVarIndex.VEL_Z: DataVarIndex.DES_VEL_Z,
        }
        
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y']

    def plot_data(self, file_path, plot_indices=None, status=None, plot_arrangement=None, special_indices=None):
        # Read the data from the csv file
        data = load_data(file_path)

        if status is not None:
            # Only plot the data that matches the status
            data = data[data[:, DataVarIndex.STATUS] == status.value]

        # Subtract the start time from the time values
        start_time = data[0, DataVarIndex.TIME]
        data[:, DataVarIndex.TIME] -= start_time
        
        # Plot the data
        if plot_indices is None: # default setting is plotting data in XYZ dimension
            plot_indices = [DataVarIndex.POS_X, DataVarIndex.POS_Y, DataVarIndex.POS_Z]
        
        num_plots = len(plot_indices) # Total number of plots

        if plot_arrangement:
            if len(plot_arrangement) != 2:
                raise ValueError("plot_arrangement must be a tuple of 2 integers")
            num_rows, num_cols = plot_arrangement
            if num_rows * num_cols <= num_plots:
                raise ValueError("plot_arrangement must have enough subplots to plot all indices")
        else:
            # Create a figure with the number of subplots equal to the number of indices
            # Find closest non-prime number to the number of plots
            if num_plots > 5:
                num_plots_non_prime = num_plots
                while is_prime(num_plots_non_prime):
                    num_plots_non_prime += 1
                
                num_plots = num_plots_non_prime
            
            # Find the closest factors of the number of plots
            for i in range(int(num_plots ** 0.5), 0, -1):
                if num_plots % i == 0:
                    num_cols = i
                    break

            num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 9))

        color_id = 0

        for plot_id, i in enumerate(plot_indices):
            color = self.colors[color_id]
            next_color = self.colors[(color_id + 1) % len(self.colors)]
            next_next_color = self.colors[(color_id + 2) % len(self.colors)]
            if isinstance(axs, np.ndarray):
                if len(axs.shape) == 1:
                    axs = axs.reshape(1, -1)
                ax = axs[plot_id // num_cols, plot_id % num_cols] if num_rows > 1 else axs[plot_id % num_cols]
            else:
                ax = axs
            if isinstance(i, tuple):
                if i[1] == "dt":
                    # Plot the actual values
                    dt = np.mean(np.diff(data[:, DataVarIndex.TIME]))
                    ax.plot(data[:-1, DataVarIndex.TIME], np.diff(data[:, i[0]]) / dt, '*-', label=f"d{i[0].name}/ dt", color=color)
                    # set axis labels
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel(f"d{i[0].name}/dt")
                else:
                    # Check if the desired value is available
                    if i[0] in self.match_desired and i[1] in self.match_desired:
                        # Plot the desired values
                        ax.plot(data[:, self.match_desired[i[0]]], data[:, self.match_desired[i[1]]], "--", color=next_color, 
                                label=self.match_desired[i[0]].name + " vs " + self.match_desired[i[1]].name)
                    
                    # Plot the actual values
                    ax.plot(data[:, i[0]], data[:, i[1]], label=f"{i[0].name} vs {i[1].name}", color=color)
                    
                    # set axis labels
                    ax.set_xlabel(i[0].name)
                    ax.set_ylabel(i[1].name)
            else:
                # Check if the desired value is available
                if i in self.match_desired:
                    # Plot the desired values
                    ax.plot(data[:, DataVarIndex.TIME], data[:, self.match_desired[i]], "--", color=next_color, 
                                label=self.match_desired[i].name)
                    
                
                "----------for test----------"
                if special_indices:
                    # Plot the actual values
                    for index, time_start in enumerate(data[:, DataVarIndex.TIME]):
                        if index % 6 == 0:  # 仅处理每隔两个的值
                            time_range_instant = np.linspace(time_start, time_start + 31/60, 31)
                            pre_state_instant = data[index, DataVarIndex.x_0:]
                            ax.plot(time_range_instant, pre_state_instant, 
                                    #label = "pre_state_inst", 
                                    color = next_next_color, 
                                    linewidth = 0.5,
                                    linestyle = '-',
                                    marker = 'o',
                                    markersize = 1.5)
                "----------for test----------"


                if i == DataVarIndex.TIME:
                    dt = np.diff(data[:, DataVarIndex.TIME])
                    ax.plot(range(len(data)-1), dt, label=i.name, color=color)

                    # set axis labels
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("dt")
                else:
                    # Plot the actual values
                    ax.plot(data[:, DataVarIndex.TIME], data[:, i], label=i.name, color=color)

                    # set axis labels
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel(i.name)

            # set the legend
            ax.legend()

            color_id = (color_id + 1) % len(self.colors)

        # Save the figure
        if self.save_fig:
            fig_name = os.path.splitext(file_path)[0] + ".png"
            ax.savefig(fig_name)

        plt.show()

if __name__ == "__main__":
    wandb_project = "test" # tac-cbf
    # Specify the indices to be plotted
    plot_indices = None
    # Specify the data by setting either the run_name or the file_name
    run_name = 'balmy-microwave-194' # run_name = 'flowing-spaceship-73'
    file_name =  None # file_name = 'data_20240604_150836_estimated_data_from_observer.csv'
    use_latest = True # use_latest has the higher periority than setting the run_name
    smoothed = False
    status = None  # Status.TRACK_TRAJ

    # Plot setting about intermediate predicted state
    plot_pred_state = False # True: plot only target state with prediction; False: plot all selected states without prediction
    special_indices = [DataVarIndex.POS_X] # Must be give in form of ndarray
    
    file_path, traj_plane = get_file_path_from_run(wandb_project, run_name, file_name, use_latest, smoothed)

    print("Plotting data from: ", file_path)
    if traj_plane is not None:
        print(f"Plotting in the {traj_plane[0]}-{traj_plane[1]}-{traj_plane[2]} plane")

    plotter = Plotter(save_fig=False)

    if plot_indices is None:
        # dictionary that maps the trajectory plane to the corresponding indices
        plane2indices_pos = {'x': DataVarIndex.POS_X, 'y': DataVarIndex.POS_Y, 'z': DataVarIndex.POS_Z}
        plane2indices_vel = {'x': DataVarIndex.VEL_X, 'y': DataVarIndex.VEL_Y, 'z': DataVarIndex.VEL_Z}
  
        # Select the indices based on the trajectory plane
        data_index_a = plane2indices_pos[traj_plane[0]]
        data_index_b = plane2indices_pos[traj_plane[1]]
        data_index_c = plane2indices_pos[traj_plane[2]]
        data_index_a_vel = plane2indices_vel[traj_plane[0]]
        data_index_b_vel = plane2indices_vel[traj_plane[1]]
        data_index_c_vel = plane2indices_vel[traj_plane[2]]

        plot_indices = [(data_index_a, data_index_b), 
                        (data_index_b, data_index_c), 
                        (data_index_a, data_index_c), 
                        data_index_a, 
                        data_index_b,
                        data_index_c,
                        data_index_a_vel,
                        data_index_b_vel,
                        data_index_c_vel,
                        DataVarIndex.ROLL,
                        DataVarIndex.PITCH,          
                        #DataVarIndex.YAW,          
                        DataVarIndex.CMD_THRUST,
                        #DataVarIndex.ROLL_RATE,
                        #DataVarIndex.YAW_RATE,
                        #DataVarIndex.PITCH_RATE,
                        ] 
        
    plotter = Plotter(save_fig=False)
    if plot_pred_state:
        plotter.plot_data(file_path, plot_indices=special_indices, status=status, special_indices=special_indices) 
    else:
        plotter.plot_data(file_path, plot_indices=plot_indices, status=status) 
