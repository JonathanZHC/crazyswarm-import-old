#!/usr/bin/env python

import json
import numpy as np
import casadi as cs
import rospy


#from position_ctl_m import PIDController, MPCController
from utils import GRAVITY, get_file_path_from_run
from helper import deg2rad, euler2quat, quat2euler, pwm2thrust, thrust2pwm

from crazyswarm.msg import StateVector, Command
from geometry_msgs.msg import Twist
# Needed to send numpy.array as a msg
from rospy.numpy_msg import numpy_msg
#from vicon_bridge.msg import Command


class Integrator:

    def __init__(self, variables, dynamics_func, delta_t):
        self.x, self.u = variables
        self.dynamics_func = dynamics_func
        self.delta_t = delta_t
        self.integrate_func = self.integrate()

    def integrate(self) -> cs.Function:
        return NotImplementedError
    
    def simulate(self, x0, u0, horizon):
        x = x0
        x_traj = [x]
        for _ in range(horizon):
            x = self.integrate_func(x, u0)
            x_traj.append(x)
        return x_traj
    

class EulerIntegrator(Integrator):

    def __init__(self, variables, dynamics_func, delta_t):
        super().__init__(variables, dynamics_func, delta_t)

    def integrate(self) -> cs.Function:
        x_next = self.x + self.delta_t * self.dynamics_func(self.x, self.u)
        return cs.Function('euler', [self.x, self.u], [x_next], ['x', 'u'], ['x_next'])
    

class RK4Integrator(Integrator):

    def __init__(self, variables, dynamics_func, delta_t):
        super().__init__(variables, dynamics_func, delta_t)

    def integrate(self) -> cs.Function:
        phi_1 = self.dynamics_func
        phi_2 = cs.Function('phi_2', [self.x, self.u], [self.dynamics_func(self.x + 0.5 * self.delta_t * phi_1(self.x, self.u), self.u)])
        phi_3 = cs.Function('phi_3', [self.x, self.u], [self.dynamics_func(self.x + 0.5 * self.delta_t * phi_2(self.x, self.u), self.u)])
        phi_4 = cs.Function('phi_4', [self.x, self.u], [self.dynamics_func(self.x + self.delta_t * phi_3(self.x, self.u), self.u)])
        rungeKutta = self.x + self.delta_t / 6 * (phi_1(self.x, self.u) + 2 * phi_2(self.x, self.u) + 2 * phi_3(self.x, self.u) + phi_4(self.x, self.u))
        return cs.Function('rk4', [self.x, self.u], [rungeKutta], ['x', 'u'], ['x_next'])

class Quad3DDynamics:

    def __init__(self, model_file):
        # Open the json file containing the model parameters
        with open(model_file) as file:
            model = json.load(file)

        # Assign the model parameters
        self.params_acc = model['params_acc']
        self.params_pitch_rate = model['params_pitch_rate']
        self.params_roll_rate = model['params_roll_rate']
        self.params_yaw_rate = model['params_yaw_rate']

        # Set the state and input dimensions
        self.state_dim = 9
        self.input_dim = 4

        # Create the symbolic variables
        self.x = cs.MX.sym('x', self.state_dim)
        self.u = cs.MX.sym('u', self.input_dim)

    def dynamics(self):
        # Unpack the state variables
        x = self.x[0]
        y = self.x[1]
        z = self.x[2]
        roll = self.x[3]
        pitch = self.x[4]
        yaw = self.x[5]
        x_dot = self.x[6]
        y_dot = self.x[7]
        z_dot = self.x[8]

        # Unpack the input variables
        cmd_roll = self.u[0]
        cmd_pitch = self.u[1]
        cmd_yaw = self.u[2]
        cmd_thrust = self.u[3]

        roll_rate = self.params_roll_rate[0] * roll + self.params_roll_rate[1] * cmd_roll
        pitch_rate = self.params_pitch_rate[0] * pitch + self.params_pitch_rate[1] * cmd_pitch
        yaw_rate = self.params_yaw_rate[0] * yaw + self.params_yaw_rate[1] * cmd_yaw

        transformed_thrust = self.params_acc[0] * cmd_thrust + self.params_acc[1]
        
        x_ddot = (cs.cos(roll) * cs.sin(pitch) * cs.cos(yaw) + cs.sin(roll) * cs.sin(yaw)) * transformed_thrust
        y_ddot = (cs.cos(roll) * cs.sin(pitch) * cs.sin(yaw) - cs.sin(roll) * cs.cos(yaw)) * transformed_thrust
        z_ddot = cs.cos(roll) * cs.cos(pitch) * transformed_thrust - GRAVITY
        

        return cs.vertcat(x_dot, y_dot, z_dot, roll_rate, pitch_rate, yaw_rate, x_ddot, y_ddot, z_ddot)
    
    def dynamics_func(self):
        return cs.Function('dynamics', [self.x, self.u], [self.dynamics()], ['x', 'u'], ['x_dot'])
    
class Simulator: # called by cf_sim.launch

    def __init__(self, cf_id_dec, sim_frequency, model_file, x0):
        self.cmd_vel_topic = cf_id_dec + "/cmd_vel"
        self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback)
        self.state_topic = f'/estimated_state'
        self.state_pub = rospy.Publisher(self.state_topic, numpy_msg(StateVector), queue_size=1)

        # Interface for receiving controller command (to be use when using estimator in simulation)
        self.sub_controller = rospy.Subscriber('/controller_command', Command)

        self.sim_frequency = sim_frequency
        self.delta_t = 1.0 / sim_frequency

        self.dynamics = Quad3DDynamics(model_file)

        self.x_cs = self.dynamics.x
        self.u_cs = self.dynamics.u
        self.dynamics_func = self.dynamics.dynamics_func()

        assert len(x0) == self.dynamics.state_dim
        self.x0 = np.array(x0, dtype=np.float64).reshape(-1, 1) 

        self.integrator = RK4Integrator([self.x_cs, self.u_cs], self.dynamics_func, self.delta_t)
        # self.integrator = EulerIntegrator([self.x_cs, self.u_cs], self.dynamics_func, self.delta_t)

        self.cmd = np.zeros(4)
        self.cmd_received = False

        self.frame_id = 0
        # Initialize the state variables
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.acc = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize rotations (quaternions).
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.omega_g = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.omega_b = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def cmd_vel_callback(self, msg):
        roll_deg = msg.linear.y
        pitch_deg = msg.linear.x
        yaw_deg = msg.angular.z
        pwm_thrust = msg.linear.z

        if pwm_thrust >= 1.0 and not self.cmd_received: 
            print("RPYT command received")
            self.cmd_received = True

        roll = deg2rad(roll_deg)
        pitch = deg2rad(pitch_deg)
        yaw = deg2rad(yaw_deg)

        thrust = pwm2thrust(pwm_thrust)

        self.cmd = np.array([roll, pitch, yaw, thrust])

    def simulate(self):
        cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust = self.cmd
        u0 = np.array([cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust])

        if self.cmd_received:
            x_traj = self.integrator.simulate(self.x0, u0, 1)
            self.x0 = np.array(x_traj[-1])

        self.pos = np.array([self.x0[0, 0], self.x0[1, 0], self.x0[2, 0]], dtype=np.float64)
        self.vel = np.array([self.x0[6, 0], self.x0[7, 0], self.x0[8, 0]], dtype=np.float64)
        
        euler = [self.x0[3, 0], self.x0[4, 0], self.x0[5, 0]]
        quat = euler2quat(euler[0], euler[1], euler[2])
        self.quat = np.array(quat, dtype=np.float64)

        state = StateVector()

        state.header.stamp.secs = rospy.Time.now().secs
        state.header.stamp.nsecs = rospy.Time.now().nsecs
        # state.header.frame_id = self.frame_id

        state.pos = self.pos
        state.vel = self.vel
        state.acc = self.acc

        state.quat = self.quat
        state.omega_g = self.omega_g
        state.omega_b = self.omega_b

        # Publish state
        self.state_pub.publish(state)

        self.frame_id += 1


def main(cf_id_dec, sim_frequency, model_file, x0):
    # Initialize the ROS node
    rospy.init_node('simulator')

    # Get the parameters from the ROS parameter server
    # cf_id_dec = rospy.get_param('~cf_id_dec')
    # control_frequency = rospy.get_param('~control_frequency')
    # model_file = rospy.get_param('~model_file')
    # x0 = rospy.get_param('~x0')

    # Create the simulator object


    simulator = Simulator(cf_id_dec, sim_frequency, model_file, x0)

    while not rospy.is_shutdown():
        simulator.simulate()
        rospy.sleep(simulator.delta_t)


if __name__ == '__main__':
    import yaml

    # change directory to the ros package directory
    import os
    base_path = os.path.dirname(os.path.realpath(__file__))

    # read config file for crayzflie id
    with open(base_path + "/../launch/crazyflies.yaml", 'r') as stream:
        try:
            cf_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        cf_id_dec = 'cf' + str(cf_config['crazyflies'][0]['id'])

    sim_freq = 200

    wandb_project = "test"
    # Get the file path from the run
    # Specify the data by setting either the run_name or the file_name
    file_name = None 
    run_name = 'major-valley-78' # easy-star-3 or major-valley-78
    use_latest = False
    smoothed = False

    batch = True # use batch identification or not: False or True
    model_file = '/home/haocheng/Experiments/figure_8/merge_model.json' # file path of identified model

    # Load path of model
    if batch == False:
        file_path, traj_plane = get_file_path_from_run(wandb_project, run_name, file_name, use_latest, smoothed)
        model_file = file_path.replace(".csv", "_model.json")

    else: # directly use pre-defined model file
        print('running case: ', model_file)


    # Define initial state (3D: 9 dimension)
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    try:
        main(cf_id_dec, sim_freq, model_file, x0)
    except rospy.ROSInterruptException:
        pass
