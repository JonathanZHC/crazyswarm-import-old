#!/usr/bin/env python

import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))

from math import pi, sin, cos
import numpy as np

from pycrazyswarm import *

def x_eval(t):
    return sin(t)
def dx_eval(t):
    return cos(t)
def ddx_eval(t):
    return -sin(t)
def y_eval(t):
    return cos(t)
def dy_eval(t):
    return -sin(t)
def ddy_eval(t):
    return -cos(t)

def executeTrajectory(timeHelper, cf, rate=100, offset=np.zeros(3)):
    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > 2*pi:
            break

        pos = np.array([x_eval(t), y_eval(t) - 1, 0.0])
        # vel = np.array([0, 0, 0])
        # acc = np.array([0, 0, 0])
        vel = np.array([dx_eval(t), dy_eval(t), 0.0])
        acc = np.array([ddx_eval(t), ddy_eval(t), 0.0])
        yaw = 0.0
        omega = np.array([0.0, 0.0, 0.0])
        
        cf.cmdFullState(
            pos + np.array(cf.initialPosition) + offset,
            vel,
            acc,
            yaw,
            omega)

        timeHelper.sleepForRate(rate)


if __name__ == "__main__":
    swarm = Crazyswarm("../../launch/crazyflies.yaml")
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    # ---- log data start
    print("press button to state sd-card logging ----- ")
    swarm.input.waitUntilButtonPressed()
    cf.setParam("usd/logging", 1)

    rate = 30.0
    Z = 0.5

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    executeTrajectory(timeHelper, cf, rate, offset=np.array([0, 0, 0.5]))

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    # ---- log data start
    print("press button to stop sd-card logging ----- ")
    swarm.input.waitUntilButtonPressed()
    cf.setParam("usd/logging", 0)