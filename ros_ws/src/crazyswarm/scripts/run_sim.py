import os
import time

from cmdVel import QuadMotion, StateEstimator
from trajectory_generator import TrajectoryGenerator2DPeriodicMotion, TrajectoryGenerator3DPeriodicMotion
from plot_data import Plotter
from utils import DataVarIndex, Status


if __name__ == "__main__":
    import wandb
    import yaml
    import json
    import rospy

    rospy.init_node('controller')

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

    # Initialize State Estimator Subscriber
    state_estimator = StateEstimator()

    # Initialize quadrotor motion
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "Experiments/figure_8_sim/")
    # create file name with date and time stamp
    file_name = "data_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
    file_path = os.path.join(data_dir, file_name)
    print("Data will be saved to: ", file_path)
    control_freq = 60.0
    sim = True
    quad_motion = QuadMotion(state_estimator, control_freq=control_freq, 
                             verbose=True, log_data=True, filename=file_path, sim=sim)

    # dictionary that maps the trajectory plane to the corresponding indices
    plane2indices_pos = {'x': DataVarIndex.POS_X, 'y': DataVarIndex.POS_Y, 'z': DataVarIndex.POS_Z}
    plane2indices_vel = {'x': DataVarIndex.VEL_X, 'y': DataVarIndex.VEL_Y, 'z': DataVarIndex.VEL_Z}
    plane2indices_acc = {'x': DataVarIndex.ACC_X, 'y': DataVarIndex.ACC_Y, 'z': DataVarIndex.ACC_Z}

    # Set parameters
    traj_type = "figure8"  # Trajectory type {"circle", "square", "figure8"}
    num_cycles = 2.0  # Number of cycles to complete
    scaling = 1.0  # Trajectory scaling
    total_time = 10.0  # Trajectory length in seconds
    sample_time = 0.01  # Sampling time, only for plotting
    traj_plane = "xyz"  # Trajectory plane
    mode = '3D' # 2D or 3D
    status = None # Status.TRACK_TRAJ

    if mode == '2D':

        # Select the indices based on the trajectory plane
        data_index_a = plane2indices_pos[traj_plane[0]]
        data_index_b = plane2indices_pos[traj_plane[1]]
        data_index_a_vel = plane2indices_vel[traj_plane[0]]
        data_index_b_vel = plane2indices_vel[traj_plane[1]]
        data_index_a_acc = plane2indices_acc[traj_plane[0]]
        data_index_b_acc = plane2indices_acc[traj_plane[1]]

        plot_indices = [(data_index_a, data_index_b), 
                        data_index_a, 
                        data_index_b,
                        DataVarIndex.PITCH,
                        data_index_a_vel,
                        data_index_b_vel,
                        DataVarIndex.CMD_THRUST,
                        DataVarIndex.PITCH_RATE,]   

        # Initialize trajectory generator
        traj = TrajectoryGenerator2DPeriodicMotion(traj_type=traj_type,
                                                    num_cycles=num_cycles,
                                                    scaling=scaling,
                                                    traj_length=total_time,
                                                    sample_time=sample_time,
                                                    traj_plane=traj_plane)

    elif mode == '3D':

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
                        DataVarIndex.PITCH, 
                        DataVarIndex.YAW,                   
                        DataVarIndex.CMD_THRUST,
                        #DataVarIndex.ROLL_RATE,
                        #DataVarIndex.YAW_RATE,
                        #DataVarIndex.PITCH_RATE,
                        ] 

        # Initialize trajectory generator
        traj = TrajectoryGenerator3DPeriodicMotion(traj_type=traj_type,
                                                    num_cycles=num_cycles,
                                                    scaling=scaling,
                                                    traj_length=total_time,
                                                    sample_time=sample_time,
                                                    traj_plane=traj_plane)

    wandb.init(project='tac-cbf', 
               config={'file_path': file_path, 
                       'control_freq': control_freq,
                       'traj_type': traj_type,
                       'num_cycles': num_cycles,
                       'scaling': scaling,
                       'traj_length': total_time,
                       'sample_time': sample_time,
                       'traj_plane': traj_plane,
                       'i_range': i_range,
                       'kp': kp,
                       'kd': kd,
                       'ki': ki,
                       'is_real': not sim,})

    # Take off
    velocity = 0.3
    height = 0.7
    target_yaw_deg = 0.0
    print("Taking off...")
    quad_motion.take_off(velocity, height, target_yaw_deg)

    # Track trajectory
    print("Tracking trajectory...")
    quad_motion.track_traj(traj)

    # Land
    print("Landing...")
    quad_motion.land(velocity)

    # finsih wandb run
    wandb.finish()

    # Plot the simulation result
    plotter = Plotter(save_fig=False)
    plotter.plot_data(file_path, plot_indices=plot_indices, status=status)



    