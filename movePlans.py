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

from myRobotSimIX import Particle_Filter

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

        ##create particle data structure
        #start pos
        #start_pos = next_waypoint
        #start angle
        #difference = [next_waypoint.real - curr_waypoint.real, next_waypoint.imag - curr_waypoint.imag]
        #start_angle = np.angle(difference[0] + difference[0]*1j) # radian

        #this will need to change in the real simulator to waypoint values
        self.robSim.pf = Particle_Filter(200 , self.robSim.posEst, self.robSim.angEst, init_pos_noise=1,init_angle_noise= np.pi/180 * 1)

        ##Loop while there are still waypoints to reach
        while len(self.sensorP.lastWaypoints[1]) > 1:
            # TODO: 
            #   1. For every iteration, we don't want to turn and move; maybe all we need is to move forward
            #   2. Following point 1, we should probably add some conditions, like we only turn iff we reach a waypoint or we drift too much
            #   3. What if we miss a waypoint?
            #   4. blocked - how to turn to avoid blocking?
            #   5. if we are really close to the target, we tend to have great angle diff, but in that way, we actually dont need to change our angle
            #   6. following point 5, we may want to turn to the desired angle when we get close to the target and turn it once for all

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
            progress("est pos:" + str(self.pos))
            progress("est ang:" + str(self.ang))
            if (numWaypoints != len(self.sensorP.lastWaypoints[1])):
                # we hit a waypoint and are heading for a new one
                progress("WAYPOINT REACHED")
                self.robSim.pf.waypoint_update(self.pos,self.ang)
                numWaypoints = len(self.sensorP.lastWaypoints[1])
            
            #position / distance
            difference = [next_waypoint[0] - self.pos.real, (next_waypoint[1] - self.pos.imag)]
            distance = linalg.norm(difference)
            #angle
            angle = np.angle(self.ang.real + self.ang.imag*1j) # radian
            target_angle = np.angle(next_waypoint[0] - self.pos.real + (next_waypoint[1] - self.pos.imag) * 1j) # radian
            turn_rads = target_angle - angle
            
            #debug
            '''
            progress("------------------------")
            progress("pos: " + str(self.pos))
            progress("target_pos: " + str(next_waypoint))
            progress("diff: " + str(difference))
            progress("dist: " + str(distance))
            progress("angle: " + str(angle))
            progress("target_ang: " + str(target_angle))
            progress("turn: " + str(turn_rads))
            progress("------------------------")
            '''
            #if we think we are at a waypoint
            min_distance_threshold = 0.01 #TODO fix this value... it should be if distance is very small... how small ... within tag?
            if(distance < min_distance_threshold):
                #TODO calculate covariance
                #TODO maybe drive straight some more first ... evaluate with testing
                #turn and move along covariance direction
                #possibly add spiral or more robust failure case
                progress("failed to reach waypoint in standard method")
                x = self.robSim.pf.particles.pos.real
                y = self.robSim.pf.particles.pos.imag
                c = np.cov(x, y)
                covariance_angle = np.angle(1 + 1j*(c[0][0]-c[0][1]))
                turn_rads = covariance_angle - angle
                
                #execute turn
                self.app.turn.ang = turn_rads
                self.app.turn.dur = 1
                self.app.turn.N = 3
                self.app.turn.start()
                yield self.forDuration(2)

                #drive back and forth some amount -- should probably be linked to particle positions

                #while waypoint number is the same
                    #drive back and forth more and more



            #default case for movement to waypoint
            else:
                ##execute turn#############################################################
                #min turn angle of 3 degrees - approx acc. of servo
                # TODO tune this value more
                min_turn_angle = 3.0 * np.pi / 180.0
                # progress(str(turn_rads))
                if(abs(turn_rads) > min_turn_angle and distance > 5):
                    progress("turn")
                    self.app.turn.ang = turn_rads
                    self.app.turn.dur = 1
                    self.app.turn.N = 3
                    self.app.turn.start()
                    yield self.forDuration(2)


                ##execute move##############################################################
                #only move by at most step_size
                #TODO tune this value
                step_size = 5
                if(distance < step_size):
                    self.app.move.dist = distance
                else:
                    self.app.move.dist = step_size

                ts,f,b = self.sensorP.lastSensor
                curr_waypoint = curr_waypoint[0] + curr_waypoint[1]*1j
                next_waypoint = next_waypoint[0] + next_waypoint[1]*1j
                self.robSim.pf.update(f, b,next_waypoint, curr_waypoint)

                # for particle in self.robSim.pf.particles:
                #     progress(str(particle.weight))

                self.app.move.dur = 4
                self.app.move.N = 5
                self.app.move.start()
                yield self.forDuration(5)
        yield
