from cmath import phase
from numpy import (
    asarray, c_, dot, isnan, append, ones, reshape, mean,
    argsort, degrees, pi, arccos, ones_like
    )
from numpy import linalg as LA
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
        self.dist = 3
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
        self.ang = 0.1
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

        #Calculate step size for absolute motion
        initAng = math.atan2(self.robSim.ang.imag, self.robSim.ang.real)
        a = self.ang-initAng
        #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
        if a > math.pi:
            a = a-2*math.pi
        elif a < -math.pi:
            a = a+2*math.pi
        deltaAng = a
        stepAng = deltaAng/float(self.N)

        for k in range(self.N):
          if not self.absolute:
              self.robSim.turn(step, self.absolute)
          else:
              self.robSim.turn(initAng+stepAng*(k+1), self.absolute)
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
        reachedFirstWaypoint = 0
        attemptsToFindWaypoint = 0
        maxAttempts = 1
        stepSize = 5
        original_dNoise = self.robSim.dNoise
        original_aNoise = self.robSim.aNoise
        while True:
            t, w = self.sensorP.lastWaypoints
            if len(w) != 0:
                break
            yield self.forDuration(0.5)
        numWaypoints = len(self.sensorP.lastWaypoints[1])
        while len(self.sensorP.lastWaypoints[1]) > 1:
            self.pos = c_[self.robSim.posEst.real, self.robSim.posEst.imag]
            self.ang = c_[self.robSim.angEst.real, self.robSim.angEst.imag]

            self.ang /= LA.norm(self.ang)
            new_time_waypoints, waypoints = self.sensorP.lastWaypoints
            curr_waypoint, next_waypoint = waypoints[0], waypoints[1]

            #We should get exactly to first waypoint in a single attempt
            if not reachedFirstWaypoint:
                self.pos = c_[self.robSim.pos.real, self.robSim.pos.imag]
                self.ang = c_[self.robSim.ang.real, self.robSim.ang.imag]
                self.robSim.posEst = self.robSim.pos
                next_waypoint = curr_waypoint[0] + curr_waypoint[1]*1j
                self.robSim.dNoise = 0
                self.robSim.aNoise = 0
                attemptsToFindWaypoint = -1
            else:
                next_waypoint = next_waypoint[0] + next_waypoint[1]*1j
                self.robSim.dNoise = original_dNoise
                self.robSim.aNoise = original_aNoise
            reachedFirstWaypoint = True

            #if we hit a waypoint move guess and noisy robot pos to waypoint
            if numWaypoints != len(self.sensorP.lastWaypoints[1]):
                way_loc = dot(curr_waypoint,[1,1j])
                self.robSim.posEst = way_loc
                numWaypoints = len(self.sensorP.lastWaypoints[1])
                attemptsToFindWaypoint = 0

            # Spiral if out of attempts
            if attemptsToFindWaypoint >= 1:
                progress('Running Spiral')
                try:
                    self.app.move.dist = stepSize if attemptsToFindWaypoint == maxAttempts else self.app.move.dist
                    self.app.turn.absolute = False
                    self.app.turn.ang = pi / 2

                    i = attemptsToFindWaypoint - maxAttempts
                    self.app.move.dur = 0.1
                    self.app.move.N = 1
                    steps = int(self.app.move.dist // 5)
                    totalDist = self.app.move.dist
                    for _ in range(steps):
                        self.app.move.dist = steps
                        self.app.move.start()
                        yield self.forDuration(0.4)
                        if (numWaypoints != len(self.sensorP.lastWaypoints[1])):
                            self.app.move.stop()
                            yield self.forDuration(1)#Wait for animation to update so robot is over waypoint
                            raise HitWaypoint("HitWaypoint")
                    self.app.move.dist = totalDist % 5
                    self.app.move.start()
                    yield self.forDuration(2)
                    self.app.move.dist = totalDist

                    self.app.turn.dur = 2
                    self.app.turn.N = 2
                    self.app.turn.start()
                    yield self.forDuration(3)
                    self.app.move.dist += stepSize * (i % 2)
                except HitWaypoint:
                    pass
            else:
                v = next_waypoint - dot(self.pos,[1,1j])
                distance = LA.norm(v)
                direction_radians = math.atan2(v.imag, v.real)

                self.app.turn.absolute = True
                self.app.turn.ang = direction_radians

                #self.app.turn.stop()
                self.app.turn.dur = 1
                self.app.turn.N = 3
                self.app.turn.start()
                yield self.forDuration(2)

                self.app.move.dist = distance
                self.app.move.dur = 4
                self.app.move.N = 5
                self.app.move.start()
                yield self.forDuration(5)
            attemptsToFindWaypoint += 1
        yield
