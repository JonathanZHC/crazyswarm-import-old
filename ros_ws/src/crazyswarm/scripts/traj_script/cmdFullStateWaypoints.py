#!/usr/bin/env python
import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))

from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pycrazyswarm import *

time = 9

def get_traj(plot=False):
# Curve fitting with waypoints.
    waypoints = [
        (0, 0, 0),
        
        (1.583, 0.084, 0.4),
        (1.783, 0.084, 0.4),

        (2.286, 2.000, 0.4),
        (2.286, 2.200, 0.4),

        (0.329, 2.878, 0.4),
        (0.129, 2.878, 0.4),

        (-1.631, 1.528, 0.4),
        (-1.831, 1.528, 0.4),

        (0, 0, 0),
    ]
    waypoints = np.array(waypoints)
    deg = 6
    t = np.arange(waypoints.shape[0])
    fit_x = np.polyfit(t, waypoints[:,0], deg)
    fit_y = np.polyfit(t, waypoints[:,1], deg)
    fit_z = np.polyfit(t, waypoints[:,2], deg)
    fx = np.poly1d(fit_x)
    fy = np.poly1d(fit_y)
    fz = np.poly1d(fit_z)
    t_scaled = np.linspace(t[0], t[-1], int(time*30))
    ref_x = fx(t_scaled)
    ref_y = fy(t_scaled)
    ref_z = fz(t_scaled)

    # Plot in 3D.
    if plot:
        ax = plt.axes(projection='3d')
        ax.plot3D(ref_x, ref_y, ref_z)
        ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
        plt.show()
        # plt.pause(2)
        # plt.close()

    return ref_x, ref_y, ref_z


def executeTrajectory(timeHelper, cf, rate=100, offset=np.zeros(3)):
    start_time = timeHelper.time()


    ref_x, ref_y, ref_z = get_traj()

    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time

        if t >= time+2:
            break
        elif t > time: 
            pos = np.array([ref_x[-1], ref_y[-1], ref_z[-1]])
            vel = np.array([0.0, 0.0, 0.0])
            acc = np.array([0.0, 0.0, 0.0])
            yaw = 0.0
            omega = np.array([0.0, 0.0, 0.0])
        elif t <= time:    
            pos = np.array([ref_x[int(t*30)], ref_y[int(t*30)], ref_z[int(t*30)]])
            vel = np.array([0.0, 0.0, 0.0])
            acc = np.array([0.0, 0.0, 0.0])
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
    # get_traj(plot=True)
    # raise
    
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