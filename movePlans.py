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

# from pyrsistent import T

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
        self.waiting_time = 4

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
        self.failure_trial = 0
        self.within_min_distnace = False
        self.failure_front_or_back = None
        self.left_back_movement_num = 16
        self.turn_rads = None
        self.curr_waypoint = None
        self.next_waypoint = None
        self.num_waypoints = 0
        self.distance = None
        self.min_distance_threshold = 5 #TODO fix this value... it should be if distance is very small... how small ... within tag?
        self.min_turn_angle = 3.0 * np.pi / 180.0
        self.step_size = 6



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

    def near_the_bound(self):
        bound1,bound2 = 110, 80
        limitation = 2
        if self.pos.real < -bound1 + limitation or bound1 - self.pos.real < limitation:
            return True
        if self.pos.imag < -bound2 + limitation or bound2 - self.pos.imag < limitation:
            return True
        return False

    def init_auto(self):
        ##get Waypoint data here
        while True:
            t, w = self.sensorP.lastWaypoints
            if len(w) != 0:
                break
            yield self.forDuration(0.5)
        self.numWaypoints = len(self.sensorP.lastWaypoints[1])

        #this will need to change in the real simulator to waypoint values
        assert(self.waypoint_to != None)
        # progress(self.waypoint_to)
        self.waypoint_to = list_to_complex(convert_waypoint(self.waypoint_to))
        self.waypoint_from = self.waypoint_to
        self.waypoint_to = list_to_complex(convert_waypoint(self.sensorP.lastWaypoints[1][1]))
        # progress(self.waypoint_from)
        self.robSim.pf = Particle_Filter(200 , self.waypoint_from, 0 + 1j, init_pos_noise=1,init_angle_noise= np.pi/180 * 1)
    
    def calculate_movement(self, print=True):
        #position / distance
        difference = [self.next_waypoint.real - self.pos.real, (self.next_waypoint.imag - self.pos.imag)]
        self.distance = linalg.norm(difference)
        #angle
        angle = np.angle(self.ang.real + self.ang.imag*1j) # radian
        target_angle = np.angle(self.next_waypoint.real - self.pos.real + (self.next_waypoint.imag - self.pos.imag) * 1j) # radian

        self.turn_rads,self.front_or_back = self.nearest_turn(angle,target_angle)
        
        progress("------------------------")
        progress("pos: " + str(self.pos))
        progress("target_pos: " + str(self.next_waypoint))
        progress("diff: " + str(difference))
        progress("dist: " + str(self.distance))
        progress("angle: " + str(angle / np.pi * 180))
        progress("target_ang: " + str(target_angle / np.pi *180))
        progress("turn: " + str(self.turn_rads /np.pi * 180))
        progress("moving torwards: " + str(self.front_or_back))
        progress("------------------------")

    def waypoint_reached(self):
        self.app.dance.start()
        yield self.forDuration(4)

        # we hit a waypoint and are heading for a new one
        self.waypoint_from = self.waypoint_to
        self.waypoint_to =  list_to_complex(convert_waypoint(self.sensorP.lastWaypoints[1][1]))
        self.curr_waypoint, self.next_waypoint = self.waypoint_from, self.waypoint_to
        progress("WAYPOINT REACHED")
        self.robSim.pf.waypoint_update(self.waypoint_from,self.ang)
        self.numWaypoints = len(self.sensorP.lastWaypoints[1])
        self.within_min_distnace = False
        self.failure_trial = 0
        self.failure_front_or_back = None
    
    def failed_to_reach_waypoint(self):
        # if(self.within_min_distnace and about_equal(f, 0.0, noise_est)):
        #TODO maybe drive straight some more first ... evaluate with testing
        #turn and move along covariance direction
        #possibly add spiral or more robust failure case
        
        progress(" \n\n FAILED to reach waypoint in standard method \n\n")
        
        if self.failure_trial % self.left_back_movement_num == 0:
            if self.failure_trial == 0:
                self.failure_front_or_back = self.front_or_back
            # move forward & left and right
            # self.robSim.liftWheels()
            # yield self.forDuration(1)
            self.app.move.dist = 10 * self.failure_front_or_back
            self.app.move.start()
            yield self.forDuration(2)

            # x = [p_i.pos.real for p_i in self.robSim.pf.particles]
            # y = [p_i.pos.imag for p_i in self.robSim.pf.particles]
            # c = np.cov(x, y)
            # covariance_angle = np.angle(1 + 1j*(c[0][0]-c[0][1]))
            # progress(covariance_angle)
            # self.turn_rads,self.front_or_back = self.nearest_turn(angle,covariance_angle)
            #execute turn
            self.app.turn.ang = np.pi /2
            self.app.turn.start()
            yield self.forDuration(2)
        # TODO: speed up back & keep record of front_or_back since it might be changing

        # near_the_bound = self.near_the_bound()
        near_the_bound = False
        # progress("NEAR THE BOUND: "+ str(near_the_bound))
        # progress("Failure_trial: "+ str(self.failure_trial))
        bf_amount = 5.0
        lr_time = self.failure_trial % self.left_back_movement_num
        if lr_time >= 0 and lr_time < self.left_back_movement_num/ 4:
            if not near_the_bound:
                self.app.move.dist = bf_amount * self.failure_front_or_back
                self.app.move.start()
                yield self.forDuration(2)
            else:
                self.failure_trial +=  2*(self.left_back_movement_num/ 4 - lr_time)
        elif lr_time >= self.left_back_movement_num/ 4 and lr_time < 3*self.left_back_movement_num/ 4:
            if not near_the_bound:
                self.app.move.dist = bf_amount * self.failure_front_or_back * -1
                self.app.move.start()
                yield self.forDuration(2)
            else:
                if lr_time > self.left_back_movement_num /2:
                    self.failure_trial +=  2*(3 * self.left_back_movement_num/ 4 - lr_time)
        elif lr_time >= 3*self.left_back_movement_num/ 4 and lr_time < self.left_back_movement_num:
            self.app.move.dist = bf_amount * self.failure_front_or_back
            self.app.move.start()
            yield self.forDuration(2)
        else:
            progress("WARNING - WRONG lr_time")

        if lr_time == self.left_back_movement_num - 1:
            self.app.turn.ang = - np.pi /2
            self.app.turn.start()
            yield self.forDuration(2)
        self.failure_trial += 1
    
    def update_pf(self):
        last_ts,f,b = self.sensorP.lastSensor
        sample_collected = [[f, b]]
        num_samples_wanted = 5
        num_sensor_trials = 0
        while (len(sample_collected) < num_samples_wanted and num_sensor_trials < 30):
            ts,f,b = self.sensorP.lastSensor
            progress("collecting sensor vals: "+ str(len(sample_collected)))
            if (ts != last_ts):
                sample_collected.append([f,b])
                last_ts = ts
            else:
                yield self.forDuration(0.2)
            num_sensor_trials +=1
        sample_collected = np.mean(np.array(sample_collected),axis = 0)
        self.robSim.pf.update(sample_collected[0], sample_collected[1], self.next_waypoint, self.curr_waypoint)

    #---------------------------------------------------------
    # NOTE: ONLY START THE AUTONOMOUS MODE WHEN WE REACH THE WAYPOINT AND TURN TO 1 + 0j
    #---------------------------------------------------------
    def behavior(self):
        yield self.init_auto()
        
        ##Loop while there are still waypoints to reach
        while len(self.sensorP.lastWaypoints[1]) > 1:

            self.curr_waypoint, self.next_waypoint = self.waypoint_from, self.waypoint_to
            self.pos, self.ang = self.robSim.pf.estimated_pose()

            if (self.numWaypoints != len(self.sensorP.lastWaypoints[1])):
                yield self.waypoint_reached()
            

            yield self.calculate_movement(print=True)
            
            if self.distance <= 1.5 or self.failure_trial > 0:
                yield self.failed_to_reach_waypoint()
            
            #default case for movement to waypoint
            else:
                if self.distance < self.min_distance_threshold:
                    self.within_min_distnace = True
                    
                elif(abs(self.turn_rads) > self.min_turn_angle and self.distance > self.min_distance_threshold):
                    # progress("turn")
                    self.app.turn.ang = self.turn_rads
                    self.app.turn.start()
                    yield self.forDuration(3)


                #only move by at most step_size
                if(self.distance < self.step_size):
                    self.app.move.dist = self.distance * self.front_or_back
                else:
                    self.app.move.dist = self.step_size * self.front_or_back

                self.app.move.start()
                yield self.forDuration(1)

                yield self.update_pf()
        yield
    # TODO: 
    #   1. For every iteration, we don't want to turn and move; maybe all we need is to move forward
    #   2. Following point 1, we should probably add some conditions, like we only turn iff we reach a waypoint or we drift too much
    #   3. What if we miss a waypoint?
    #   4. blocked - how to turn to avoid blocking?
    #   5. if we are really close to the target, we tend to have great angle diff, but in that way, we actually dont need to change our angle
    #   6. following point 5, we may want to turn to the desired angle when we get close to the target and turn it once for all