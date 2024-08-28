#!/usr/bin/env python
import os, sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../'))

from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pycrazyswarm import *

time = 15

def get_traj(plot=False):
# Curve fitting with waypoints.
    waypoints = [
        (0, 0, 0.5),
        (1.2, 0.5, 0.5),
        (1.8, 0.5, 0.5),
        (3.0, 1.2, 0.5),
        (3.0, 1.8, 0.5),
        (1.3, 3.0, 0.5),
        (0.7, 3.0, 0.5),
        (0.5, 4.2, 0.5),
        (0.5, 4.8, 0.5),
        (0.5, 5.5, 0.5),
    ]
    waypoints = np.array(waypoints)
    waypoints = waypoints + np.array([-1, -3, 0.0])
    print(waypoints)
    deg = 5
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
    # print(ref_x)
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
        
        # print(pos + np.array(cf.initialPosition) + offset)
        cf.cmdFullState(
            pos + offset,
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
    # print("press button to takeoff ----- ")
    # swarm.input.waitUntilButtonPressed()

    rate = 30.0
    Z = 1.0

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    executeTrajectory(timeHelper, cf, rate, offset=np.array([0, 0, 0.5]))

    # timeHelper.sleep(17)
    # print("press button to return...")
    # swarm.input.waitUntilButtonPressed()

    cf.goTo(np.array([-0.5,  2.5, 1.75]), 0, 2.5)
    timeHelper.sleep(3.0)
    cf.goTo(np.array([-1.0, -3.0, 1.75]), 0, 6.0) 
    timeHelper.sleep(7.0)

    # print("press button to land...")
    # swarm.input.waitUntilButtonPressed()

    cf.land(targetHeight=0.03, duration=Z+2.0)
    timeHelper.sleep(Z+2.0)
