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
    return sin(t)*cos(t)
def dy_eval(t):
    return cos(2*t)
def ddy_eval(t):
    return -sin(2*t)

def executeTrajectory(timeHelper, cf, rate=100, offset=np.zeros(3)):
    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > 2*pi + 1.5:
            break
        elif t > 2*pi: 
            pos = np.array([0.0, 0.0, 0.0])
            vel = np.array([0.0, 0.0, 0.0])
            acc = np.array([0.0, 0.0, 0.0])
            yaw = 0.0
            omega = np.array([0.0, 0.0, 0.0])

        else:
            pos = np.array([x_eval(t), y_eval(t), 0.0])
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
    '''
    Parameter equations for figure 8 
    x = a*sin(t)
    y = a*sint(t)*cost(t)

    Taylor expansion of trig functions 
    cos(x) ~ 1 - x**2/2! + x**4/4! - x**6/6! + ...
    sin(x) ~ x - x**3/3! + x**5/5! - x**7/7! + ...

    x = a*()
    '''
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