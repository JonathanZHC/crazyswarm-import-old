import numpy as np
#import scipy
#import math
import timeit
import rospy
import json
import os

#from helper import rotation2euler, euler2rotation, clamp
from casadi import SX, vertcat, sin, cos
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


class MPCModel:
    def __init__(self):
        '''Basic parameter setting'''
        # Define name of solver to be used in script
        model_name = 'tracking_mpc'

        # Define Gravitational Acceleration
        GRAVITY = 9.806

        # Import model parameters
        model_file = '/home/haocheng/Experiments/figure_8/merge_model.json'
        with open(model_file) as file:
            identified_model = json.load(file)

        params_acc = identified_model['params_acc']
        params_pitch_rate = identified_model['params_pitch_rate']
        params_roll_rate = identified_model['params_roll_rate']
        params_yaw_rate = identified_model['params_yaw_rate']
    
        # Define dimension of state, input and output vector
        # State vector: [px, py, pz, vx, vy, vz, r, p, y]
        # Input vector: [f_cmd, r_cmd, p_cmd, y_cmd]
        # Output vector: [px, py, pz, y]
        self.dim_state_position = 3
        self.dim_state_velocity = 3
        self.dim_state_euler = 3
        self.dim_state = self.dim_state_position + self.dim_state_velocity + self.dim_state_euler
        self.dim_output = 4
        self.dim_input = 4

        '''Model setting'''
        # define basic variables in state and input vector
        px = SX.sym('px')
        py = SX.sym('py')
        pz = SX.sym('pz')
        vx = SX.sym('vx')
        vy = SX.sym('vy')
        vz = SX.sym('vz')
        r = SX.sym('r')
        p = SX.sym('p')
        y = SX.sym('y')
        f_collective = SX.sym('f_collective')
        r_cmd = SX.sym('r_cmd')
        p_cmd = SX.sym('p_cmd')
        y_cmd = SX.sym('y_cmd')

        # define state and input vector
        states = vertcat(px, py, pz, vx, vy, vz, r, p, y)
        inputs = vertcat(f_collective, r_cmd, p_cmd, y_cmd)

        # Define nonlinear system dynamics
        f = vertcat(vx, 
                    vy, 
                    vz, 
                    (params_acc[0] * f_collective + params_acc[1]) * sin(p),
                    -(params_acc[0] * f_collective + params_acc[1]) * sin(y) * cos(p),
                    (params_acc[0] * f_collective + params_acc[1]) * cos(y) * cos(p) - GRAVITY,
                    params_roll_rate[0][0] * r + params_roll_rate[1][0] * r_cmd,
                    params_pitch_rate[0][0] * p + params_pitch_rate[1][0] * p_cmd,
                    params_yaw_rate[0][0] * y + params_yaw_rate[1][0] * y_cmd)

        # Initialize the nonlinear model for NMPC formulation
        self.model = AcadosModel()
        self.model.name = model_name
        self.model.f_expl_expr = f
        self.model.f_impl_expr = None
        self.model.x = states
        self.model.u = inputs


class MPCSolver:
    def __init__(self, conrtol_freq):

        # Read parameters file for the controller gains
        script_dir = os.path.dirname(__file__)
        param_file = os.path.join(script_dir, "parameters.json")
        with open(param_file) as f:
            params = json.load(f)  

        # Define prediction horizon and time
        self.freq = conrtol_freq
        N = params['MPC_solver']['N'] # Prediction Horizon
        T = N / self.freq # Prediction Time
    
        # Define weight matrix in cost function
        Q = params['MPC_solver']['Q'] # Weight for state
        R = params['MPC_solver']['R'] # Weight for input

        # Create as object of pre-defined model class
        model_obj = MPCModel()

        '''MPC Problem setting'''
        # Initialize the solver
        ocp = AcadosOcp()
        ocp.model = model_obj.model
        self.dim_state = model_obj.dim_state
        self.dim_input = model_obj.dim_input
        self.dim_output = model_obj.dim_output
    
        # Define time parameters
        ocp.dims.N = N
        ocp.solver_options.tf = T

        # Define charactoristics of MPC solver
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # Define other hyperparameters in SQP solving
        ocp.solver_options.nlp_solver_max_iter = 500
        #ocp.solver_options.nlp_solver_tol_stat = 1E-3
        #ocp.solver_options.nlp_solver_tol_eq = 1E-3

        '''Cost function setting'''
        # Set type of cost function
        ocp.cost.cost_type = 'LINEAR_LS' # Stage cost 
        ocp.cost.cost_type_e = 'LINEAR_LS' # Terminal cost 
    
        # Define weight matrix in cost function
        # Initialize weight matrix for stage cost
        W = np.zeros((model_obj.dim_output + model_obj.dim_input, model_obj.dim_output + model_obj.dim_input))
        W[:model_obj.dim_output, :model_obj.dim_output] = np.eye(model_obj.dim_output) * Q
        W[model_obj.dim_output:, model_obj.dim_output:] = np.eye(model_obj.dim_input) * R

        # Define weight matrix for stage and terminal cost
        ocp.cost.W = W # Stage cost 
        ocp.cost.W_e = np.eye(model_obj.dim_output) * Q  # Terminal cost 

        # Initialize output function for stage cost
        # Define output matrix
        Vx = np.zeros((model_obj.dim_output + model_obj.dim_input, model_obj.dim_state))
        Vx[:model_obj.dim_state_position, :model_obj.dim_state_position] = np.eye(model_obj.dim_state_position)
        Vx[model_obj.dim_state_position, -3] = np.eye(1)
        ocp.cost.Vx = Vx
        # Define breakthrough matrix
        Vu = np.zeros((model_obj.dim_output + model_obj.dim_input, model_obj.dim_input)) 
        Vu[model_obj.dim_output:, :] = np.eye(model_obj.dim_input)
        ocp.cost.Vu = Vu

        # Initialize output function for terminal cost
        # Define output matrix
        Vx_e = np.zeros((model_obj.dim_output, model_obj.dim_state))
        Vx_e[:model_obj.dim_state_position, :model_obj.dim_state_position] = np.eye(model_obj.dim_state_position)
        Vx_e[model_obj.dim_state_position:model_obj.dim_output, (model_obj.dim_state_position - model_obj.dim_output):] = np.eye(model_obj.dim_output - model_obj.dim_state_position)
        ocp.cost.Vx_e = Vx_e

        '''Constraints setting'''
        # Input Constraints
        #ocp.constraints.idxbu = np.array(range(model_obj.dim_input))
        #ocp.constraints.lbu = np.array([-1.0])
        #ocp.constraints.ubu = np.array([1.0])
        # State Constraints
        ocp.constraints.idxbx = np.array(range(model_obj.dim_state)) # must be defined before upper/lower limit
        ocp.constraints.lbx = np.array(params['MPC_problem']['lbx'])
        ocp.constraints.ubx = np.array(params['MPC_problem']['ubx'])
        # Define initial state for problem solving
        ocp.constraints.x0 = np.array(params['MPC_problem']['x0'])
        
        '''Reference value setting'''
        # Initialize reference points for stage and terminal cost
        # For yref, the first 'dim_output' variables will be updated (reference state) in each step, 
        # when the next 'dim_input' variables keep zero (reference input: 0) 
        ocp.cost.yref = np.zeros(model_obj.dim_output + model_obj.dim_input)
        ocp.cost.yref_e = np.zeros(model_obj.dim_output)

        '''MPC Solver initiallizing'''
        # Initialize the MPC solver
        self.ocp = ocp
        self.model_name = 'tracking_mpc'
        self.solver = AcadosOcpSolver(self.ocp, json_file=f'{self.model_name}.json')



class PositionController:
    def __init__(self, quad=None, conrtol_freq=30):
        if not quad is None:
            # get parameters
            self.params = quad.params
            self.controller_params = quad.controller

        self.i_error = 0

        # abc-formula coefficients for thrust to pwm conversion
        # pwm = a * thrust^2 + b * thrust + c
        self.a_coeff = -1.1264
        self.b_coeff = 2.2541
        self.c_coeff = 0.0209
        self.pwm_max = 65535.0
        
        # Create an object of pre-defined Solver class
        self.solver_obj = MPCSolver(conrtol_freq)

        # Initialize parameters to be used in MPC solver
        self.MPC_N = self.solver_obj.ocp.dims.N
        self.MPC_dim_state = self.solver_obj.dim_state
        self.MPC_dim_input = self.solver_obj.dim_input
        self.MPC_dim_output = self.solver_obj.dim_output

        # Internal parameter for warm starting
        self.prev_solution_x = np.zeros((self.MPC_N, self.MPC_dim_state))
        self.prev_solution_u = np.zeros((self.MPC_N, self.MPC_dim_input))

        # Parameters for time recording
        self.counter = int(0)
        self.ave_cycle_time = float(0.0)
        self.max_cycle_time = float(0.0)
    
    def thrust2pwm(self, thrust):
        """Convert thrust to pwm using a quadratic function."""
        pwm = self.a_coeff * thrust * thrust + self.b_coeff * thrust + self.c_coeff
        pwm = np.maximum(pwm, 0.0)
        pwm = np.minimum(pwm, 1.0)
        thrust_pwm = pwm * self.pwm_max
        return thrust_pwm
    
    def pwm2thrust(self, pwm):
        """Convert pwm to thrust using a quadratic function."""
        pwm_scaled = pwm / self.pwm_max
        # solve quadratic equation using abc formula
        thrust = (-self.b_coeff + np.sqrt(self.b_coeff**2 - 4 * self.a_coeff * (self.c_coeff - pwm_scaled))) / (2 * self.a_coeff)
        return thrust

    def mpc_controller(self, current_state, target_state):
        # Set initial state
        self.solver_obj.solver.set(0, "lbx", current_state)
        self.solver_obj.solver.set(0, "ubx", current_state)
    
        # Set tracking reference
        yref = np.zeros(self.MPC_dim_output + self.MPC_dim_input)
        yref[:self.MPC_dim_output] = target_state
        self.solver_obj.solver.set(self.MPC_N, "yref", target_state)
        for i in range(self.MPC_N):
            self.solver_obj.solver.set(i, "yref", yref)
        
        # Warm starting: initialize a policy for SQP
        for i in range(self.MPC_N):
            self.solver_obj.solver.set(i, "u", self.prev_solution_u[i])
            self.solver_obj.solver.set(i, "x", self.prev_solution_x[i])
    
        # Solve MPC problem
        start = timeit.default_timer() # To record cycle time for each iteration
        status = self.solver_obj.solver.solve()
        if status != 0:
            raise Exception(f"ACADOS failed to solve a feasible solution，return status: {status}")

        # Calculate average cycle time
        cur_cycle_time = timeit.default_timer() - start
        total_time = self.ave_cycle_time * self.counter + cur_cycle_time
        self.counter += 1
        self.ave_cycle_time = total_time / self.counter
        rospy.loginfo('Cumulative average cycle time: %f' %self.ave_cycle_time)
        self.max_cycle_time = max(self.max_cycle_time, cur_cycle_time)
        rospy.loginfo('Cumulative maximal cycle time: %f' %self.max_cycle_time)

        # save for warm starting
        for i in range(self.MPC_N): 
            self.prev_solution_u[i] = self.solver_obj.solver.get(i, "u")
            self.prev_solution_x[i] = self.solver_obj.solver.get(i, "x")
        
        # get optimal policy and return as new input
        u_opt = self.solver_obj.solver.get(0, "u")

        return u_opt

    def compute_action(self, measured_pos, measured_rpy, measured_vel, desired_pos, desired_roll):
        """Compute the thrust and euler angles for the drone to reach the desired position.
        
        Args:
            measured_pos (np.array): current position of the drone
            measured_rpy (np.array): current roll, pitch, yaw angles of the drone in radians
            desired_pos (np.array): desired position of the drone
            desired_roll (float): desired roll angle of the drone in radians
            dt (float): time step
            
        Returns:
            thrust_desired (float): desired thrust
            euler_desired (np.array): desired euler angles
        """

        rospy.loginfo("measured_rpy: ")
        rospy.loginfo(measured_rpy)
        rospy.loginfo("desired_roll ")
        rospy.loginfo(desired_roll)

        current_state = np.hstack((measured_pos, measured_vel, measured_rpy))
        target_output = np.hstack((desired_pos, desired_roll))

        # Call API function for Acados to solve MPC problem on current time step
        input_desired = self.mpc_controller(current_state, target_output)

        current_thrust = input_desired[0]
        euler_desired = input_desired[1:]
        
        # Check on current thrust
        current_thrust = max(current_thrust, 0.3 * self.params.quad.m * self.params.quad.g) # incase too small deceleration
        current_thrust = min(current_thrust, 1.8 * self.params.quad.m * self.params.quad.g) # incase too large acceleration
        
        # Transform thrust_desired back into pwm_desired
        pwm_desired = self.thrust2pwm(current_thrust)

        return pwm_desired, euler_desired

    def position_controller_reset(self):
        self.i_error = np.zeros(3)


