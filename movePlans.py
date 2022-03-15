from cmath import phase
from pickle import TRUE
from turtle import back
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

from particleFilter import Particle_Filter, about_equal

from waypointShared import lineSensorResponse, lineDist

# VERY IMPORTANT ########################################################################################################
#determines if we are running real or simulated robot
real_robot = True
############################################################################################################



#converts a waypoint coordinate - list format into complex number format
def list_to_complex(waypoint):
    return waypoint[0] + 1j* waypoint[1]


#accounts for coordinate flipping when using the real camera system
def convert_waypoint(waypoint):
    #flip x and y if real robot
    if real_robot:
        return [waypoint[1], waypoint[0]]
    else:
        return waypoint

# HitWaypoint Exception
class HitWaypoint(Exception):
    pass

class MoveDistClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.dist = 5
        self.waiting_time = 1

    def behavior(self):
        self.robSim.move(self.dist)
        yield self.forDuration(self.waiting_time)

class LiftWheelsClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.waiting_time = 1
    def behavior(self):
        self.robSim.liftWheels()
        yield self.forDuration(self.waiting_time)

class TurnClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.ang = 0.05
        self.absolute = False
        self.waiting_time = 1

    def behavior(self):
        yield self.robSim.turn(self.ang)
        yield self.forDuration(self.waiting_time)

class DanceClass(Plan):
    def __init__(self, app, robSim):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.waiting_time = 1

    def behavior(self):
        self.robSim.dance()
        yield self.forDuration(self.waiting_time)

class Auto(Plan):
    """
    Plan takes control of the robot and navigates the waypoints.
    """
    def __init__(self, app, robSim, sensorP):#, pfP):
        Plan.__init__(self, app)
        self.robSim = robSim
        self.sensorP = sensorP
        self.pos = [0, 0]
        self.front_or_back = True
        self.waypoint_from = None
        self.waypoint_to = None

    def nearest_turn(self,curr_ang,target_ang):
        front_near_angle = target_ang - curr_ang
        if abs(front_near_angle) > np.pi:
            front_near_angle =  (2 * np.pi - abs(front_near_angle) ) * -1 * np.sign(front_near_angle)
        back_angle = 0
        if curr_ang > 0:
            back_angle = curr_ang - np.pi
        else:
            back_angle = curr_ang + np.pi
        back_near_angle = target_ang - back_angle
        if abs(back_near_angle) > np.pi:
            back_near_angle =  (2 * np.pi - abs(back_near_angle) ) * -1 * np.sign(back_near_angle)
        if abs(front_near_angle) > abs(back_near_angle):
            return back_near_angle,-1
        else:
            return front_near_angle,1


    def behavior(self):
        #---------------------------------------------------------
        # NOTE: ONLY START THE AUTONOMOUS MODE WHEN WE REACH THE WAYPOINT AND TURN TO 1 + 0j
        #---------------------------------------------------------
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
        assert(self.waypoint_to != None)
        # progress(self.waypoint_to)
        self.waypoint_to = list_to_complex(convert_waypoint(self.waypoint_to))
        self.waypoint_from = self.waypoint_to
        self.waypoint_to = list_to_complex(convert_waypoint(self.sensorP.lastWaypoints[1][1]))
        # progress(self.waypoint_from)
        self.robSim.pf = Particle_Filter(200 , self.waypoint_from, 0 + 1j, init_pos_noise=1,init_angle_noise= np.pi/180 * 1)
        self.within_min_distnace = False
        failure_trial = 0
        failure_front_or_back = None
        left_back_movement_num = 16
        ##Loop while there are still waypoints to reach
        while len(self.sensorP.lastWaypoints[1]) > 1:
            # TODO: 
            #   1. For every iteration, we don't want to turn and move; maybe all we need is to move forward
            #   2. Following point 1, we should probably add some conditions, like we only turn iff we reach a waypoint or we drift too much
            #   3. What if we miss a waypoint?
            #   4. blocked - how to turn to avoid blocking?
            #   5. if we are really close to the target, we tend to have great angle diff, but in that way, we actually dont need to change our angle
            #   6. following point 5, we may want to turn to the desired angle when we get close to the target and turn it once for all

            curr_waypoint, next_waypoint = self.waypoint_from, self.waypoint_to
            self.pos, self.ang = self.robSim.pf.estimated_pose()

            if (numWaypoints != len(self.sensorP.lastWaypoints[1])):
                self.app.dance.start()
                yield self.forDuration(2)

                # we hit a waypoint and are heading for a new one
                self.app.move.dist = 8 * self.front_or_back
                self.app.move.start()
                yield self.forDuration(2)

                self.waypoint_from = self.waypoint_to
                self.waypoint_to =  list_to_complex(convert_waypoint(self.sensorP.lastWaypoints[1][1]))
                curr_waypoint, next_waypoint = self.waypoint_from, self.waypoint_to
                progress("WAYPOINT REACHED")
                self.robSim.pf.waypoint_update(self.waypoint_from,self.ang)
                numWaypoints = len(self.sensorP.lastWaypoints[1])
                self.within_min_distnace = False
                failure_trial = 0
                failure_front_or_back = None
            
            #position / distance
            difference = [next_waypoint.real - self.pos.real, (next_waypoint.imag - self.pos.imag)]
            distance = linalg.norm(difference)
            #angle
            angle = np.angle(self.ang.real + self.ang.imag*1j) # radian
            target_angle = np.angle(next_waypoint.real - self.pos.real + (next_waypoint.imag - self.pos.imag) * 1j) # radian

            turn_rads,self.front_or_back = self.nearest_turn(angle,target_angle)
            
            progress("------------------------")
            progress("pos: " + str(self.pos))
            progress("target_pos: " + str(next_waypoint))
            progress("diff: " + str(difference))
            progress("dist: " + str(distance))
            progress("angle: " + str(angle / np.pi * 180))
            progress("target_ang: " + str(target_angle / np.pi *180))
            progress("turn: " + str(turn_rads /np.pi * 180))
            progress("moving torwards: " + str(self.front_or_back))
            progress("------------------------")


            #if we think we are at a waypoint
            min_distance_threshold = 5 #TODO fix this value... it should be if distance is very small... how small ... within tag?
            
            #we think we are close to waypoint and then see that we pass the end of the line
            ts,f,b = self.sensorP.lastSensor
            noise_est = 10.0
            
            if distance <= 1.5 or failure_trial > 0:
            # if(self.within_min_distnace and about_equal(f, 0.0, noise_est)):
                #TODO maybe drive straight some more first ... evaluate with testing
                #turn and move along covariance direction
                #possibly add spiral or more robust failure case
                
                progress(" \n\n\n\n FAILED to reach waypoint in standard method \n\n\n\n")
                
                if failure_trial % left_back_movement_num == 0:
                    failure_front_or_back = self.front_or_back
                    # move forward & left and right
                    self.robSim.liftWheels()
                    yield self.forDuration(1)
                    self.app.move.dist = 10 * failure_front_or_back
                    self.app.move.start()
                    yield self.forDuration(2)

                    # x = [p_i.pos.real for p_i in self.robSim.pf.particles]
                    # y = [p_i.pos.imag for p_i in self.robSim.pf.particles]
                    # c = np.cov(x, y)
                    # covariance_angle = np.angle(1 + 1j*(c[0][0]-c[0][1]))
                    # progress(covariance_angle)
                    # turn_rads,self.front_or_back = self.nearest_turn(angle,covariance_angle)
                    #execute turn
                    self.app.turn.ang = np.pi /2
                    self.app.turn.start()
                    yield self.forDuration(2)
                # TODO: speed up back & keep record of front_or_back since it might be changing
                bf_amount = 5.0
                lr_time = failure_trial % left_back_movement_num
                if lr_time >= 0 and lr_time < left_back_movement_num/ 4:
                    self.app.move.dist = bf_amount * failure_front_or_back
                    self.app.move.start()
                    yield self.forDuration(1)
                elif lr_time >= left_back_movement_num/ 4 and lr_time < 3*left_back_movement_num/ 4:
                    self.app.move.dist = bf_amount * failure_front_or_back * -1
                    self.app.move.start()
                    yield self.forDuration(1)
                elif lr_time >= 3*left_back_movement_num/ 4 and lr_time < left_back_movement_num:
                    self.app.move.dist = bf_amount * failure_front_or_back
                    self.app.move.start()
                    yield self.forDuration(1)
                else:
                    progress("WARNING - WRONG lr_time")

                if lr_time == left_back_movement_num - 1:
                    self.app.turn.ang = - np.pi /2
                    self.app.turn.start()
                    yield self.forDuration(2)
                failure_trial += 1
            #default case for movement to waypoint
            else:
                #min turn angle of 3 degrees - approx acc. of servo
                # TODO tune this value more
                min_turn_angle = 3.0 * np.pi / 180.0

                if distance < min_distance_threshold:
                    self.within_min_distnace = True
                    
                elif(abs(turn_rads) > min_turn_angle and distance > min_distance_threshold):
                    # progress("turn")

                    #debug 
                    # progress("------------------------")
                    # progress("pos: " + str(self.pos))
                    # progress("target_pos: " + str(next_waypoint))
                    # progress("diff: " + str(difference))
                    # progress("dist: " + str(distance))
                    # progress("angle: " + str(angle / np.pi * 180))
                    # progress("target_ang: " + str(target_angle / np.pi *180))
                    # progress("turn: " + str(turn_rads /np.pi * 180))
                    # progress("moving torwards: " + str(self.front_or_back))
                    # progress("------------------------")

                    self.app.turn.ang = turn_rads
                    self.app.turn.start()
                    yield self.forDuration(2)


                #only move by at most step_size
                #TODO tune this value
                step_size = 10
                if(distance < step_size):
                    self.app.move.dist = distance * self.front_or_back
                else:
                    self.app.move.dist = step_size * self.front_or_back

                #ts,f,b = self.sensorP.lastSensor
                self.robSim.pf.update(f, b, next_waypoint,curr_waypoint)

                # for particle in self.robSim.pf.particles:
                #     progress(str(particle.weight))

                self.app.move.start()
                yield self.forDuration(2)
        yield
