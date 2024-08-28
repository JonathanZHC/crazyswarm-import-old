# general
import numpy as np
import json
from munch import Munch

# utilities
from helper import update_full_state

class quadrotor():
    def __init__(self):
        '''
        Initialize simulator
        '''

        # load parameters
        self.params = self.get_params()

        # initialize quadrotor state
        self.state = self.initialize_zero_state()

        # initialize controller
        self.controller = self.initialize_controller()

        # initialize reference structure (what is sent to the drone)
        self.reference = self.initialize_zero_state()

        # initialize desired structure (what is desired to track)
        self.desired = self.initialize_zero_state()

    def get_params(self):
        '''
        Load parameters from parameters.json

        :return:
            params - dictionary with simulation, quadrotor physics, and controller parameters
        '''

        # load parameters and convert to . dict format
        load_params = json.load(open("parameters.json"))
        params = Munch.fromDict(load_params)

        return params

    def initialize_zero_state(self):
        '''
        Initialize quadrotor state

        :return:
        state - dictionary containing quadrotor state
        '''

        # initialize state with zero pose
        state = {
            "x" : 0.0,
            "y" : 0.0,
            "z" : 0.0,

            "vx" : 0.0,
            "vy" : 0.0,
            "vz" : 0.0,

            "ax" : 0.0,
            "ay" : 0.0,
            "az" : 0.0,

            "roll"  : 0.0,
            "pitch" : 0.0,
            "yaw"   : 0.0,

            "p" : 0.0,
            "q" : 0.0,
            "r" : 0.0,

            "R" : np.eye(3),
            "quat" : np.array([1.0, 0.0, 0.0, 0.0])
        }

        # convert dict to . dict format
        state = Munch.fromDict(state)

        # construct full state array
        state.full_state = update_full_state(state)

        return state

    def initialize_controller(self):
        '''
        Initialize controller

        :return:
        controller - a dictionary containing data for computing control command
        '''

        # initialize controller
        controller = {
            "pos_ctl" : {
                "i_error" : np.zeros(3)
            },

            "att_ctl" : {
                "i_error" : np.zeros(3),
                "prev_pqr" : np.zeros(3),
                "prev_pqr_des" : np.zeros(3)
            }
        }

        # convert dict to . dict format
        controller = Munch.fromDict(controller)

        return controller