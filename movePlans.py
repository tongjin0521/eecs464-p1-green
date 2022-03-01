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

        ##get Waypoint data here
        while True:
            t, w = self.sensorP.lastWaypoints
            if len(w) != 0:
                break
            yield self.forDuration(0.5)
        numWaypoints = len(self.sensorP.lastWaypoints[1])

        ##Loop while there are still waypoints to reach
        while len(self.sensorP.lastWaypoints[1]) > 1:
            # TODO: 
            #   1. For every iteration, we don't want to turn and move; maybe all we need is to move forward
            #   2. Following point 1, we should probably add some conditions, like we only turn iff we reach a waypoint or we drift too much
            #   3. What if we miss a waypoint?
            #   4. blocked - how to turn to avoid blocking?
            
            ##old version using basic state estimate
            ''' 
            ##fetch position and angle estimates as well as waypoint locations
            self.pos = c_[self.robSim.posEst.real, self.robSim.posEst.imag]
            self.ang = c_[self.robSim.angEst.real, self.robSim.angEst.imag]
            new_time_waypoints, waypoints = self.sensorP.lastWaypoints
            curr_waypoint, next_waypoint = waypoints[0], waypoints[1]

            ##compute angle and direction to turn and move
            difference = next_waypoint - self.pos
            distance = linalg.norm(difference)
            angle = np.angle(self.robSim.angEst.real + self.robSim.angEst.imag*1j) # radian
            target_angle = np.angle(difference[0][0] + difference[0][1]*1j) # radian
            turn_rads = target_angle - angle
            '''
            ##new version using PF state estimate
            new_time_waypoints, waypoints = self.sensorP.lastWaypoints
            curr_waypoint, next_waypoint = waypoints[0], waypoints[1]

            self.pos, self.ang = self.robSim.pf.estimated_pose()
            #position / distance
            difference = next_waypoint - self.pos
            distance = linalg.norm(difference)
            #angle
            target_angle = np.angle(difference[0][0] + difference[0][1]*1j) # radian
            turn_rads = target_angle - self.ang

            #if we think we are at a waypoint
            min_distance_threshold = 0.01 #TODO fix this value... it should be if distance is very small... how small ... within tag?
            if(distance < min_distance_threshold):
                #TODO calculate covariance 
                #turn and move along this direction
                #possibly add spiral or more robust failure case
                progress("failed to reach waypoint in standard method")

            #default case for movement to waypoint
            else:
                ##execute turn#############################################################
                #min turn angle of 3 degrees - approx acc. of servo
                # TODO tune this value more
                min_turn_angle = 3.0 * 3.14159 / 180.0
                progress(str(turn_rads))
                if(abs(turn_rads) > min_turn_angle):
                    progress("turn")
                    self.app.turn.ang = turn_rads
                    self.app.turn.dur = 1
                    self.app.turn.N = 3
                    self.app.turn.start()
                    yield self.forDuration(2)


                ##execute move##############################################################
                #only move by at most step_size
                step_size = 5
                if(distance < step_size):
                    self.app.move.dist = distance
                else:
                    self.app.move.dist = step_size

                self.app.move.dur = 4
                self.app.move.N = 5
                self.app.move.start()
                yield self.forDuration(5)
        yield
