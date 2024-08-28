#!/usr/bin/env python
import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))

import numpy as np

from pycrazyswarm import *

waypoints = [[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0]]
s_per_length = 4
hover_time = 1

def executeTrajectory(timeHelper, cf, rate=100, offset=np.zeros(3)):
    start_time = timeHelper.time()
    idx = 0
    starting_pos = np.array([0, 0, 0])
    flag = False
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > s_per_length*(idx+1):
            if not flag:
                flag = True
                starting_pos = starting_pos + np.array(waypoints[idx])
                idx += 1
                idx = min(3, idx)
            else:
                flag = False
        elif t > s_per_length*(idx+1) - hover_time:
            cf.cmdFullState(
                np.array(waypoints[idx]) + np.array(cf.initialPosition) + offset + starting_pos,
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                0.0,
                np.array([0.0, 0.0, 0.0]))
        else:
            cf.cmdFullState(
                (t - s_per_length*idx)/(s_per_length-hover_time)*np.array(waypoints[idx]) + np.array(cf.initialPosition) + offset + starting_pos,
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                0.0,
                np.array([0.0, 0.0, 0.0]))
            
        if t > s_per_length*4:
            break

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

    executeTrajectory(timeHelper, cf, rate, offset=np.array([0, 0, 0.42]))

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    # ---- log data start
    print("press button to stop sd-card logging ----- ")
    swarm.input.waitUntilButtonPressed()
    cf.setParam("usd/logging", 0)