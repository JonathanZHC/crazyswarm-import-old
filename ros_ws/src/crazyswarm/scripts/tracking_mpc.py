import numpy as np
import json

from casadi import SX, vertcat, sin, cos
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


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
dim_state_position = 3
dim_state_velocity = 3
dim_state_euler = 3
dim_state = dim_state_position + dim_state_velocity + dim_state_euler
dim_output = 4
dim_input = 4

# Define prediction horizon and time
freq = 60
N = 60 # Prediction Horizon
T = N / freq # Prediction Time

# Define weight matrix in cost function
Q = 0.1 # Weight for state
R = 0.1 # Weight for input



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
model = AcadosModel()
model.f_expl_expr = f
model.f_impl_expr = None
model.x = states
model.u = inputs
model.name = model_name



'''MPC Solver setting'''
# Initialize the solver
ocp = AcadosOcp()
ocp.model = model

# Define time parameters
ocp.dims.N = N
ocp.solver_options.tf = T

# Define charactoristics of MPC solver
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP_RTI'

# Define other hyperparameters in SQP solving
ocp.solver_options.nlp_solver_max_iter = 500
ocp.solver_options.nlp_solver_tol_stat = 1E-3
ocp.solver_options.nlp_solver_tol_eq = 1E-3



'''Cost function setting'''
# Set type of cost function
ocp.cost.cost_type = 'LINEAR_LS' # Stage cost 
ocp.cost.cost_type_e = 'LINEAR_LS' # Terminal cost 


# Define weight matrix in cost function
# Initialize weight matrix for stage cost
W = np.zeros((dim_output + dim_input, dim_output + dim_input))
W[:dim_output, :dim_output] = np.eye(dim_output) * Q
W[dim_output:, dim_output:] = np.eye(dim_input) * R

# Define weight matrix for stage and terminal cost
ocp.cost.W = W # Stage cost 
ocp.cost.W_e = np.eye(dim_output) * Q  # Terminal cost 


# Initialize reference points for stage and terminal cost
# For yref, the first 'dim_output' variables will be updated (reference state) in each step, 
# when the next 'dim_input' variables keep zero (reference input: 0) 
ocp.cost.yref = np.zeros(dim_output + dim_input)
ocp.cost.yref_e = np.zeros(dim_output)


# Initialize output function for stage cost
# Define output matrix
Vx = np.zeros((dim_output + dim_input, dim_state))
Vx[:dim_state_position, :dim_state_position] = np.eye(dim_state_position)
Vx[dim_state_position:dim_output, (dim_state_position - dim_output):] = np.eye(dim_output - dim_state_position)
ocp.cost.Vx = Vx
# Define breakthrough matrix
Vu = np.zeros((dim_output + dim_input, dim_input)) 
Vu[dim_output:, :] = np.eye(dim_input)
ocp.cost.Vu = Vu

# Initialize output function for terminal cost
# Define output matrix
Vx_e = np.zeros((dim_output, dim_state))
Vx_e[:dim_state_position, :dim_state_position] = np.eye(dim_state_position)
Vx_e[dim_state_position:dim_output, (dim_state_position - dim_output):] = np.eye(dim_output - dim_state_position)
ocp.cost.Vx_e = Vx_e



'''Constraint setting'''
'''
# 设置状态和控制输入约束
ocp.constraints.x0 = np.array([0.0, 0.0])
ocp.constraints.lbu = np.array([-1.0])
ocp.constraints.ubu = np.array([1.0])
ocp.constraints.idxbu = np.array([0])
'''



# 导出C代码
ocp.code_export_directory = '/home/haocheng/crazyswarm-import/ros_ws/src/crazyswarm/externalDependencies/libmpc'

# 生成并导出Python接口
acados_solver = AcadosOcpSolver(ocp, json_file=f'{model_name}.json')

print(f"ACADOS代码生成完成，已导出到当前目录下")
