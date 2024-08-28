import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd,'../traj_script/sim_trajs/competition/'))
import pickle as pkl

pkl_file = cwd + "/cmd_test1.pkl"


with open(pkl_file, 'rb') as f_handle:
    cmd_log = pkl.load(f_handle)

print(cmd_log)