from cmath import phase
from numpy import (
    asarray, c_, dot, isnan, append, ones, reshape, mean,
    argsort, degrees, pi, arccos, ones_like
    )
from numpy import linalg
import numpy as np
import math

try:
    from joy.plans import Plan
except ImportError:
    import sys, os
    sys.path.append(os.path.expanduser('~/pyckbot/hrb/'))
    from joy.plans import Plan

from joy import progress

from waypointShared import lineSensorResponse, lineDist

# HitWaypoint Exception
class HitWaypoint(Exception):
    pass

class MoveDistClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        # Distance to travel
        self.dist = 2
        # Duration of travel [sec]
        self.dur = 2
        # Number of intermediate steps
        self.N = 5

    def behavior(self):
        # Compute step along the forward direction
        step = self.dist / float(self.N)
        # print('Step size: ', step)
        dt = self.dur / float(self.N)
        for k in range(self.N):
          self.robSim.move(step)
          #print(self.robSim.pos)
          yield self.forDuration(dt)

class LiftWheelsClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.direction = 1
    def behavior(self):
        yield self.robSim.liftWheels()

class TurnClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        # Angle to turn [rad]
        self.ang = 0.05
        # Duration of travel [sec]
        self.dur = 1.0
        # Number of intermediate steps
        self.N = 3
        # Abs loc
        self.absolute = False

    def behavior(self):
        # Compute rotation step for relative motion
        dt = self.dur / float(self.N)
        step = self.ang / float(self.N)

        for k in range(self.N):
            self.robSim.turn(step)
            yield self.forDuration(dt)

class Auto(Plan):
    """
    Plan takes control of the robot and navigates the waypoints.
    """
    def __init__(self, app, robSim, sensorP):#, pfP):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.sensorP = sensorP
        self.pos = [0, 0]

    def behavior(self):
        # Move and Turn Fast/In 1 Step
        self.app.move.N = 1
        self.app.move.dur = 0.5
        self.app.turn.N = 1
        self.app.turn.dur = 0.0
        stepSize = 5

        ##get Waypoint data here
        while True:
            t, w = self.sensorP.lastWaypoints
            if len(w) != 0:
                break
            yield self.forDuration(0.5)
        numWaypoints = len(self.sensorP.lastWaypoints[1])


        while len(self.sensorP.lastWaypoints[1]) > 1:

            self.pos = c_[self.robSim.posEst.real, self.robSim.posEst.imag]
            self.ang = c_[self.robSim.angEst.real, self.robSim.angEst.imag]
            
            new_time_waypoints, waypoints = self.sensorP.lastWaypoints
            curr_waypoint, next_waypoint = waypoints[0], waypoints[1]

            progress( "Waypoints: " + str(waypoints))
            progress("position: " + str(self.pos))
            progress("angle: " + str(self.ang))

            difference = next_waypoint - self.pos
            distance = linalg.norm(difference)

            # angle = np.angle(self.ang)[0][0]
            # target_angle = np.angle(difference)[0][0]
            angle = np.angle(self.robSim.angEst.real + self.robSim.angEst.imag*1j)
            target_angle = np.angle(difference[0][0] + difference[0][1]*1j)
            progress("differnece: " + str(difference[0][0] + difference[0][1]*1j))
            progress("current angle: " + str(angle))
            progress("desired angle: " + str(target_angle))

            turn_rads = target_angle - angle

            progress("amount to turn: ", str(turn_rads))

            self.app.turn.ang = turn_rads
            self.app.turn.dur = 1
            self.app.turn.N = 3
            self.app.turn.start()
            yield self.forDuration(2)
            progress("angle: " + str(np.angle(self.robSim.angEst.real + self.robSim.angEst.imag*1j)))


            self.app.move.dist = distance
            self.app.move.dur = 4
            self.app.move.N = 5
            self.app.move.start()
            yield self.forDuration(5)

        # next_waypoint = next_waypoint[0] + next_waypoint[1]*1j
        # v = next_waypoint - dot(self.pos,[1,1j])
        # distance = linalg.norm(v)
        # direction_radians = math.atan2(v.imag, v.real)
        

        # self.app.turn.ang = direction_radians + self.robSim.angEst

        # self.app.turn.dur = 1
        # self.app.turn.N = 3
        # self.app.turn.start()
        # yield self.forDuration(2)








        # while len(self.sensorP.lastWaypoints[1]) > 1:
        #     self.pos = c_[self.robSim.posEst.real, self.robSim.posEst.imag]
        #     self.ang = c_[self.robSim.angEst.real, self.robSim.angEst.imag]

        #     self.ang /= linalg.norm(self.ang)
        #     new_time_waypoints, waypoints = self.sensorP.lastWaypoints
        #     curr_waypoint, next_waypoint = waypoints[0], waypoints[1]
            
        #     next_waypoint = next_waypoint[0] + next_waypoint[1]*1j
        #     v = next_waypoint - dot(self.pos,[1,1j])

        #     distance = linalg.norm(v)
        #     direction_radians = math.atan2(v.imag, v.real)

        #     #if we hit a waypoint move guess and noisy robot pos to waypoint
        #     if numWaypoints != len(self.sensorP.lastWaypoints[1]):
        #         way_loc = dot(curr_waypoint,[1,1j])
        #         self.robSim.posEst = way_loc
        #         numWaypoints = len(self.sensorP.lastWaypoints[1])
            
        #     self.app.turn.absolute = True
        #     self.app.turn.ang = direction_radians

        #     #self.app.turn.stop()
        #     self.app.turn.dur = 1
        #     self.app.turn.N = 3
        #     self.app.turn.start()
        #     yield self.forDuration(2)

        #     self.app.move.dist = distance
        #     self.app.move.dur = 4
        #     self.app.move.N = 5
        #     self.app.move.start()
        #     yield self.forDuration(5)
        yield
