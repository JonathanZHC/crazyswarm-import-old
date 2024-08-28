"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) cmdSimOnly (optional)
        4) interStepLearn (optional)
        5) interEpisodeLearn (optional)

"""
import numpy as np
import sys
from collections import deque

try:
    sys.path.append('sim_trajs/competition')
    from competition_utils import Command, PIDController, timing_step, timing_ep
except ImportError:
    # Test import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep


def catmul_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
               t: float) -> np.ndarray:
    """Calculate the Catmul-Rom spline parameterized by p0-p3 at timestep t.

    Args:
        p0: First parameterization point. Determines the initial direction of the spline.
        p1: Second parameterization point. Starting point of the spline.
        p2: Third parameterization point. End point of the spline.
        p3: Fourth parameterization point. Determines the end direction of the spline.
        t: Time parameterization. Must be in [0, 1].

    Returns:
        The point on the spline at time t.
    """
    assert 0 <= t <= 1, "Time parameter must be in [0, 1]"
    x0 = t * ((2 - t) * t - 1) * p0
    x1 = (t * t * (3 * t - 5) + 2) * p1
    x2 = t * ((4 - 3 * t) * t + 1) * p2
    x3 = (t - 1) * t * t * p3
    return 0.5 * (x0 + x1 + x2 + x3)


def catmul_spline(points: np.ndarray, num_samples: int = 100) -> np.ndarray:
    """Calculate a Catmul-Rom spline passing through all points and sample it evenly.

    Args:
        points: Array of points where each point is a coordinate that the spline passes through
        num_samples: Number of interpolation samples.

    Returns:
        The interpolated points as 2D numpy array.
    """
    assert isinstance(points, np.ndarray), "Points must be a numpy array"
    assert points.ndim == 2, "Point array does not match the expected dimension"
    assert len(points) > 1, "Spline needs at least two points"
    result = []
    div = len(points) - 1
    samples = [num_samples // div + (1 if x < num_samples % div else 0) for x in range(div)]
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        for j in range(samples[i]):
            t = j / samples[i]
            result.append(catmul_rom(p0, p1, p2, p3, t))
    return np.array(result)


class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and conrol parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialized simple PID Controller.
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        waypoints = []
        for g in self.NOMINAL_GATES:
            waypoints.append(g[:3])
        waypoints.append([
            initial_info["x_reference"][0], initial_info["x_reference"][2],
            initial_info["x_reference"][4]
        ])
        # We set the initial position at runtime when we know where exactly the drone is located
        self.waypoints = np.array(waypoints)
        self.duration = 15
        takeoff_pos = np.array([[-1, -2, 0.5]])  # Position in the air
        waypoints = np.concatenate((takeoff_pos, self.waypoints[1:, :]), axis=0)
        # The reference trajectory is calculated when the drone has taken off, taking its position
        # in the air as trajectory start to prevent discontinuities
        self.ref_x = None
        self.ref_y = None
        self.ref_z = None
        # Controller helpers
        self.takeoff = False
        self.takeoff_cmd = False
        self.control_transfer = False
        self.tstart = None

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this function to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError(
                "[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False."
            )

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Heuristic solution to solve the environment.
        if not self.takeoff:  # Take-off first
            self.takeoff = obs[4] > 0.8
            height = 1
            duration = 2
            command_type = Command(2)  # Take-off cmd
            args = [height, duration]
            if self.takeoff_cmd:
                command_type = Command(0)  # None.
                args = []
            else:
                self.takeoff_cmd = True
        elif not self.control_transfer:  # Transfer the control to low-level cmd after takeoff
            self.tstart = time  # Start the trajectory timer from liftoff on
            # Calculate the trajectory from the take-off point
            num_samples = int(self.duration * self.CTRL_FREQ)
            takeoff_pos = np.array([[obs[0], obs[2], obs[4]]])  # Position in the air
            waypoints = np.concatenate((takeoff_pos, self.waypoints), axis=0)
            trajectory = catmul_spline(waypoints, num_samples=num_samples)
            self.ref_x = trajectory[:, 0]
            self.ref_y = trajectory[:, 1]
            self.ref_z = trajectory[:, 2]
            self.control_transfer = True
            command_type = Command(6)  # Notify setpoint stop to transfer to low-level control
            args = []
        elif (step := int((time - self.tstart) * self.CTRL_FREQ)) < len(self.ref_x):
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)
            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self, time, obs, reward=None, done=None, info=None):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            Re-implement this function to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError(
                "[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True."
            )

        iteration = int(time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if iteration < len(self.ref_x):
            target_p = np.array(
                [self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)

        #########################
        # REPLACE THIS (END) ####
        #########################

        return target_p, target_v

    @timing_step
    def interStepLearn(self, action, obs, reward, done, info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1

        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
