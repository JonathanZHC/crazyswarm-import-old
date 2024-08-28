#!/usr/bin/env python
import numpy as np
import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))

from pycrazyswarm import *

def executeTrajectory(timeHelper, cf, rate=100, offset=np.zeros(3)):
    start_time = timeHelper.time()

    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        cf.cmdFullState(
            np.array([0.0, 0.0, 0.0]) + np.array(cf.initialPosition) + offset,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            0.0,
            np.array([0.0, 0.0, 0.0]))

        if t > 10:
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
    Z = 2.0

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    executeTrajectory(timeHelper, cf, rate, offset=np.array([0, 0, 2.0]))

    cf.notifySetpointsStop()
    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    # ---- log data start
    print("press button to stop sd-card logging ----- ")
    swarm.input.waitUntilButtonPressed()
    cf.setParam("usd/logging", 0)