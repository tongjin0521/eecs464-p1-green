# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 20:31:13 2014

@author: shrevzen-home
"""
from cmath import pi
from functools import total_ordering
from pickle import FALSE, TRUE
import sys, os
from unittest import result
if 'pyckbot/hrb/' not in sys.path:
    sys.path.append(os.path.expanduser('~/pyckbot/hrb/'))

from gzip import open as opengz
from json import dumps as json_dumps
import numpy as np
from numpy.linalg import svd
from numpy.random import randn
from waypointShared import lineDist

from pdb import set_trace as DEBUG
import math
import random

DEFAULT_MSG_TEMPLATE = {
    0 : [[2016, 1070], [1993, 1091], [2022, 1115], [2044, 1093]],
    1 : [[1822, 1323], [1824, 1287], [1787, 1281], [1784, 1315]],
    2 : [[1795, 911], [1766, 894], [1749, 916], [1779, 933]],
    3 : [[1451, 876], [1428, 896], [1454, 917], [1476, 896]],
    4 : [[1374, 1278], [1410, 1268], [1399, 1236], [1364, 1243]],
    22 : [[1744, 622], [1743, 646], [1774, 650], [1774, 626]],
    23 : [[2274, 1171], [2312, 1177], [2306, 1146], [2271, 1141]],
    24 : [[1100, 975], [1110, 946], [1077, 938], [1066, 966]],
    25 : [[1666, 1629], [1665, 1589], [1625, 1585], [1626, 1624]],
    26 : [[2305, 1663], [2310, 1704], [2352, 1708], [2345, 1667]],
    27 : [[2230, 697], [2230, 721], [2262, 727], [2260, 704]],
    28 : [[911, 1525], [952, 1523], [953, 1483], [913, 1486]],
    29 : [[1222, 542], [1193, 537], [1186, 558], [1216, 566]],
}
try:
  from randArenaOutput import MSG_TEMPLATE, randSeed
  print("### Using randomly generated arena from seed %d" % randSeed)
except:
  MSG_TEMPLATE = DEFAULT_MSG_TEMPLATE
  print("### Using DEFAULT arena")

# NOTE: must be AFTER randAreaOutput import
from waypointShared import waypoints, corners, ROBOT_TAGID, ref, fitHomography, Sensor
from robotSimIX import RobotSimInterface

#returns if a is equal to b within noise margins
def about_equal(a, b, noise):
    if((a <= (b + noise)) and (a >= (b - noise))):
        return True
    else:
        return False

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


####class that represents a particle
class Particle:
    def __init__(self, pos, angle, weight):
        self.pos = pos
        self.angle = angle
        self.weight = weight

class Particle_Filter:


    # NOTE: we initialize when we reach the first waypoint and have turned to the second waypoint
    def __init__(self, num_particles, init_pos, init_angle,init_pos_noise = 1,init_angle_noise = np.pi / 180 * 5):
        self.dist_noise = init_pos_noise
        self.angle_noise = init_angle_noise

        self.Sensor = Sensor()
        
        assert(num_particles > 1)
        self.num_particles = num_particles
        self.particles = []
        for i in range(self.num_particles):
            init_pos_noise_i = (randn()+1j*randn())*init_pos_noise
            init_angle_noise_i = np.exp(1j*randn()*init_angle_noise)
            self.particles.append(Particle(pos = init_pos + init_pos_noise_i,angle = init_angle *init_angle_noise_i , weight = 1/ self.num_particles) )

    def move_update(self,move_dist):
        move_dist_noise = 0.5
        move_angle_noise = np.pi / 180 * 0.5

        for particle_i in self.particles:
            ## NOTE: add noise as dist increases (if needed)
            move_dist_noise_i = (randn()+1j*randn())*move_dist_noise
            move_angle_noise_i = np.exp(1j*randn()*move_angle_noise)
            particle_i.pos +=  (particle_i.angle * move_angle_noise_i ) * move_dist + move_dist_noise_i
            particle_i.angle *= move_angle_noise_i
        
    def turn_update(self,turn_angle):
        turn_dist_noise = 0.001
        turn_angle_noise = np.pi / 180 * 1

        for particle_i in self.particles:
            ## NOTE: add noise as angle increases (if needed)
            turn_dist_noise_i = (randn()+1j*randn())*turn_dist_noise
            turn_angle_noise_i = randn()*turn_angle_noise
            particle_i.pos +=  turn_dist_noise_i
            particle_i.angle *=  np.exp(1j*(turn_angle + turn_angle_noise_i))

    def waypoint_update(self,waypoint_pos,waypoint_angle):
        assert(len(self.particles) == self.num_particles)
        waypoint_dist_noise = self.dist_noise
        waypoint_angle_noise = self.angle_noise
        previous_particles_num = int(self.num_particles/10 * 2)

        self.particles.sort(key= lambda x: x.weight , reverse= True)        
        new_particles_num = self.num_particles - previous_particles_num
        self.particles = self.normalized_particles(self.particles[:previous_particles_num])
        for i in range(new_particles_num):
            waypoint_pos_noise_i = (randn()+1j*randn())*waypoint_dist_noise
            waypoint_angle_noise_i = np.exp(1j*randn()*waypoint_angle_noise)
            self.particles.append(Particle(pos = waypoint_pos + waypoint_pos_noise_i,angle = waypoint_angle *waypoint_angle_noise_i , weight = 1/ self.num_particles) )
        assert(len(self.particles) == self.num_particles)

    def resample_particles(self):
        curr_pos,curr_ang = self.estimated_pose()
        assert(len(self.particles) == self.num_particles)
        resample_pos_noise = self.dist_noise
        resample_angle_noise = self.angle_noise
        num_new_particles = int( 0.4 * self.num_particles)
        num_old_particles = self.num_particles - num_new_particles
        result_particles = []
        r = random.random() * 1 / self.num_particles
        c = self.particles[0].weight
        i = 0
        for m in range(1,num_new_particles+1):
            u = r + (m - 1)* 1 / self.num_particles
            # print(u,c,m,i)
            while ( u > c):
                i += 1
                c += self.particles[i].weight
            result_particles.append(Particle(pos = self.particles[i].pos, angle= self.particles[i].angle, weight = 1 / self.num_particles))

        for m in range(num_old_particles):
            resample_pos_noise_i = (randn()+1j*randn())*resample_pos_noise
            resample_angle_noise_i = np.exp(1j*randn()*resample_angle_noise)
            result_particles.append(Particle(pos = curr_pos + resample_pos_noise_i,angle = curr_ang *resample_angle_noise_i , weight = 1/ self.num_particles) )  
        self.particles = result_particles
        assert(len(self.particles) == self.num_particles)

    def line_dist(self,c,a,b):
        p = np.array([c.real,c.imag])
        lp1 = np.array([a.real,a.imag])
        lp2 = np.array([b.real,b.imag])
        var1 = lp2[1] - lp1[1]
        var2 = lp1[0] - lp2[0]
        var3 = (lp1[1] - lp2[1]) * lp1[0] + (lp2[0] - lp1[0]) * lp1[1]
        res = np.abs(var1 * p[0] + var2 * p[1] + var3) / np.sqrt(var1**2 + var2**2)
        res0 = 1/(1+res**2)
        # res1 = np.asarray(res0.clip(0,0.9999)*256, np.uint8)
        return res0

    def update(self, measured_f, measured_b, a, b):
        print("updating")
        print("f: " + str(measured_f))
        print("b: " + str(measured_b))
        noise_est = 2.0
        ##discard distance measurements if they are about 0
        if(about_equal(measured_f, 0.0, noise_est) and about_equal(measured_b, 0.0, noise_est)):
            return
        elif(about_equal(measured_f, 0.0, noise_est)):
            real_dist = measured_b
        elif(about_equal(measured_b, 0.0, noise_est)):
            real_dist = measured_f
        else:
            real_dist = (measured_f + measured_b)/2
        
        #for particle in self.particles:
        for i in range(0, len(self.particles)):
            particle_distance = self.Sensor.sense(None, self.particles[i].pos, a, b) ##TODO
            normalized_distance = abs(particle_distance - real_dist) / 255.0
            scale_factor = (1 - normalized_distance)
            if(about_equal(real_dist, particle_distance, noise_est)):
                self.particles[i].weight *= 1
            else:
                self.particles[i].weight *= scale_factor
        self.particles = self.normalized_particles(self.particles)
        print(str(max(x.weight for x in self.particles)))



    def sensor_update(self,sensor,last_waypoint,next_waypoint,REF_TO_CAMERA):
        assert(len(self.particles) == self.num_particles)
        last_waypoint = last_waypoint[0] + last_waypoint[1]*1j
        next_waypoint = next_waypoint[0] + next_waypoint[1]*1j
        ts,f,b = sensor.lastSensor
        max_weight = 0
        sensor_sum = f + b
        for i in range(self.num_particles):
            # if i %20 ==0:
            #     print(self.particles[i].pos)
            #     print(self.particles[i].pos - self.particles[i].angle * 0.1)
            #     print(self.particles[i].pos + self.particles[i].angle * 0.1)
            # TODO: TAG LENGTH 
            estimated_f = self.particles[i].pos - self.particles[i].angle * 10
            estimated_b = self.particles[i].pos + self.particles[i].angle * 10
            # coordinates = np.asarray([[estimated_f.real,estimated_f.imag],[estimated_b.real,estimated_b.imag],next_waypoint,last_waypoint])
            # xy1 = np.c_[coordinates, np.ones_like(coordinates[:,1])]
            # xy1 = np.dot(xy1,REF_TO_CAMERA)
            # rec_centered_camera_coordinates  = (xy1[:,:2]/xy1[:,[2]])
            # print(rec_centered_camera_coordinates.shape)
            # estimated_f = rec_centered_camera_coordinates[0,0] + rec_centered_camera_coordinates[0,1] * 1j
            # estimated_b = rec_centered_camera_coordinates[1,0] + rec_centered_camera_coordinates[1,1] * 1j
            # cam_next_waypoint = rec_centered_camera_coordinates[2,0] + rec_centered_camera_coordinates[2,1] * 1j
            # cam_last_waypoint = rec_centered_camera_coordinates[3,0] + rec_centered_camera_coordinates[3,1] * 1j
            # print(estimated_f,estimated_b,cam_next_waypoint,cam_last_waypoint)
            estimated_f = self.line_dist(estimated_f,last_waypoint,next_waypoint)
            estimated_b = self.line_dist(estimated_b,last_waypoint,next_waypoint)
            # estimated_sum = estimated_f + estimated_b
            # estimated_f = estimated_f / (estimated_sum) * sensor_sum
            # estimated_b = estimated_b / (estimated_sum) * sensor_sum
            # if i % 20 == 0:
            print(f,b)
            print(estimated_f,estimated_b)
            print("--------")
            self.particles[i].weight = np.sqrt((estimated_f - f)**2 + (estimated_b - b )**2)
        self.particles = self.normalized_particles(self.particles)
        assert(len(self.particles) == self.num_particles)

    def normalized_particles(self,unnomarlized_particles):
        total_weight = 0
        for particle_i in unnomarlized_particles:
            total_weight += particle_i.weight
        for particle_i in unnomarlized_particles:
            particle_i.weight *= len(unnomarlized_particles) * 1 / (self.num_particles * total_weight)
        return unnomarlized_particles

    def estimated_pose(self):
        needed_particles_num = int(self.num_particles/10)

        particles_copy = self.particles.copy()
        particles_copy.sort(key= lambda x: x.weight , reverse= True)
        needed_particles = particles_copy[:needed_particles_num]
        estimated_pos = 0 + 0j
        estimated_ang = 0 + 0j
        needed_total_weight = 0
        for particle_i in needed_particles:
            estimated_pos += particle_i.pos * particle_i.weight
            estimated_ang += particle_i.angle * particle_i.weight
            needed_total_weight += particle_i.weight
        return estimated_pos/needed_total_weight, estimated_ang/needed_total_weight
    
   

class RobotSim( RobotSimInterface ):
    def __init__(self, app=None, sensor = None, *args, **kw):
        RobotSimInterface.__init__(self, *args, **kw)
        self.app = app
        self.sensor = sensor
        self.sensor_ts = -1
        #self.real_distance_noise = 0.1 # Distance noise - likely higher magnitude
        #self.real_angle_noise = 0.02 # Angle noise - difference in angle 
        #self.real_base_noise = 0.01 - additional base noise for slipping or other movement
        
        #motor accuraccy is based on encoders about 3 deg acc best case
        #noise in distance, then angle, then distance
        #reset isnt exact at waypoint, it can anywhere within the tag
        #we do have a state estimate that can be displayed but it is currently commented out


        #real noises used to update real robot posisiton
        self.real_distance_noise = 0.00001 # Distance noise
        self.real_angle_noise = 0.00001 # Angle noise
        self.real_stopping_noise = 0.00001 #this will update angle when stopping
        self.real_base_noise = 0.00001 #base noise
        

        #Handle coordiante transformation
        ## Reverse Homography
        roi  = np.array([ np.mean(MSG_TEMPLATE[nmi],0) for nmi in corners ] )
        roi  = np.c_[roi, np.ones_like(roi[:,1])]
        self.CAMERA_TO_REF = fitHomography(roi, ref)
        self.REF_TO_CAMERA = np.linalg.inv(self.CAMERA_TO_REF)
        
        #Convert tag pos to ref coordinates
        xy1 = np.c_[self.tagPos, np.ones_like(self.tagPos[:,1])]
        xy1 = np.dot(xy1, self.CAMERA_TO_REF)
        self.tagPosRef = (xy1[:,:2]/xy1[:,[2]])

        # Model the tag
        #Converts tagPos to complex numbers. tag is a 4x1 array of the corners of the robot tag in complex numbers
        tag = np.dot(self.tagPosRef,[1,1j])
        self.zTag = tag-np.mean(tag)
        self.pos = np.mean(tag)
        self.ang = 1+0j
        self.baseAng = self.ang
        self.wheelsDown = True
        self.laserBlocked = False

        #Create Particles to hold our estimates of our position
        self.tagPosRefEst = self.tagPosRef
        self.posEst = self.pos
        self.angEst = self.ang
    
        self.pf = Particle_Filter(10 ,self.pos,self.ang,init_pos_noise=1,init_angle_noise= np.pi/180 * 1)


        rec_dim = [25, 10] # height, width
        base_dim = [18, 12]
        scale_f = 1/2
        self.rec_unrotated = np.dot([[1,scale_f/2*1j],
                                    [1,-scale_f/2*1j],
                                    [-1,-scale_f *1j],
                                    [-1,scale_f *1j],
                                    [1,scale_f/2*1j]],
                                    rec_dim)
        self.base_unrotated = np.dot([[1,1j],
                                    [1,-1j],
                                    [-1,-1j],
                                    [-1,+1j],
                                    [1,+1j]],
                                    base_dim)

    def move(self,dist):
        if not self.wheelsDown:
            self.liftWheels()
        # Move in direction of self.ang
        #  If we assume Gaussian errors in velocity, distance error will grow
        #  as sqrt of goal distance
        noise = (randn()+1j*randn())*self.real_distance_noise*np.sqrt(abs(dist))
        self.pos += self.ang * dist + noise
        #noise = randn()*self.real_angle_noise
        #self.ang *= np.exp(1j*(noise))
        self.pf.move_update(dist)
        self.posEst += self.angEst * dist

    def turn(self,ang):
        if self.wheelsDown:
            self.liftWheels()
        # Turn by ang (plus noise)
        noise = randn()*self.real_angle_noise
        self.ang *= np.exp(1j*(ang + noise))
        self.pf.turn_update(ang)
        self.angEst *= np.exp(1j*(ang))
        
    def liftWheels(self):
        #whenver wheels are lifted, add noise to the base angle
        noise = randn()*self.real_base_noise
        self.baseAng *=  np.exp(1j*noise)
        #self.ang *= self.baseAng
        self.wheelsDown = not self.wheelsDown

    def plot_rect(self,coordinates,plt = "~plot",color = 'r',ls='-',both= False):
        # Tranfsorm to camera coordiantes
        xy1 = np.c_[coordinates, np.ones_like(coordinates[:,1])]
        xy1 = np.dot(xy1,self.REF_TO_CAMERA)
        rec_centered_camera_coordinates  = (xy1[:,:2]/xy1[:,[2]])
        x = [int(x) for x in rec_centered_camera_coordinates[:,0]]
        y = [int(y) for y in rec_centered_camera_coordinates[:,1]]
        if plt == "~plot":
            if both:
                self.visRobot(plt, x, y, c=color,linestyle=ls)
            self.visArena(plt, x, y, c=color,linestyle=ls, alpha=.5)
        elif plt == "~scatter":
            if both:
                self.visRobot(plt, x, y, c=color)
            self.visArena(plt, x, y, c=color)
        else:
            print("WARNING - WRONG PLOT STYLE")

    def refreshState(self):
        # update particles
        
        # ts,f,b = self.sensor.lastSensor
        # if f!=None and b!=None and self.sensor_ts!= ts and f * b > 0 and ((f+b) > 10) :
        #     self.sensor_ts = ts
        #     self.pf.resample_particles()
        #     new_time_waypoints, waypoints = self.sensor.lastWaypoints
        #     self.pf.sensor_update(self.sensor,waypoints[0],waypoints[1],self.REF_TO_CAMERA)

        # Compute tag points relative to tag center, with 1st point on real axis
        tag = self.zTag * self.ang + self.pos
        tagEst = self.zTag * self.angEst + self.posEst
        #tagEst = self.zTag
        # New tag position is set by position and angle
        self.tagPosRef = np.c_[tagEst.real,tagEst.imag]
        #In order for waypontTask.py to work, self.tagPos needs to be in camera coordiantes
        xy1 = np.c_[self.tagPosRef, np.ones_like(self.tagPosRef[:,1])]
        xy1 = np.dot(xy1,self.REF_TO_CAMERA)
        self.tagPos = (xy1[:,:2]/xy1[:,[2]])

        # Laser axis is based on tag and turret
        c = np.mean(tag) # center
        cEst = np.mean(tagEst) # center
        c_camera = np.mean(self.tagPos)

        '''
        #Get current position in reference coordinates
        pnt = array([c.real, c.imag, 1.0])
        xy1 = dot(pnt, self.CAMERA_TO_REF)
        xy1_trans = (xy1[:2]/xy1[2]).T
        robotPosRefComplex = xy1_trans[0]+xy1_trans[1]*1j
        '''

        #mean distance of tag corners from tag center
        r = np.mean(abs(tag-c))
        #Laser points 90 degrees counter clockwise from base angle
        ax = self.baseAng*(0+1j)

        #Check if laser blocked
        laserAngleRad = math.atan2(ax.imag, ax.real)
        robotAngleRad = math.atan2(self.ang.imag, self.ang.real)
        tolerance = 10*(math.pi/180)
        if robotAngleRad>(laserAngleRad-tolerance) and robotAngleRad<(laserAngleRad+tolerance):
            self.laserBlocked = True
            laserFormat = '--'
        elif (robotAngleRad+math.pi)>(laserAngleRad-tolerance) and (robotAngleRad+math.pi)<(laserAngleRad+tolerance):
            self.laserBlocked = True
            laserFormat = '--'
        else:
            self.laserBlocked = False
            laserFormat = '-'

        self.laserAxis = [[c.real, c.imag],[(c+ax).real, (c+ax).imag]]
        ## Example of visualization API
        # Visualize laser
        vl = c + np.asarray([0,ax*100*r])
        #Convert vl to camera coordinates
        vl = np.asarray([[x.real, x.imag] for x in vl])
        xy1 = np.c_[vl ,np.ones_like(vl[:,1])]
        xy1 = np.dot(xy1,self.REF_TO_CAMERA)
        vl_camera  = (xy1[:,:2]/xy1[:,[2]])
        # Start a new visual in robot subplot
        self.visRobotClear()
        # plot command in robot subplot,
        #   '~' prefix changes coordinates using homography
        self.visRobot('~plot',
            [int(v) for v in vl_camera[:,0]],
            [int(v) for v in vl_camera[:,1]],
            c='g', linestyle=laserFormat)
        # Start a new visual in arena subplot
        self.visArenaClear()
        # plot command in arena subplot,
        #   '~' prefix changes coordinates using homography
        self.visArena('~plot',
            [int(v) for v in vl_camera[:,0]],
            [int(v) for v in vl_camera[:,1]],
            c='g',alpha=0.5, linestyle=laserFormat)

        # We can call any plot axis class methods, e.g. grid
        self.visArena('grid',1)

        ## TODO: Simplify into function? Or make math easier?

        ## ------real robot------ dashed lines
        #self.plot_rect(np.asarray([[x.real, x.imag] for x in self.rec_unrotated*self.angEst+cEst]),plt = "~plot", color = 'b',ls="--", both = False)
        #self.plot_rect(np.asarray([[x.real, x.imag] for x in self.base_unrotated*self.baseAng + cEst]),plt = "~plot", color='g',ls = "--", both = False)

        ## ------PF robot------ real lines 
        #self.plot_rect(np.asarray([[x.pos.real,x.pos.imag] for x in self.pf.particles]),plt="~scatter", color = 'darkseagreen',both = True)
        self.plot_rect(np.asarray([[x.real, x.imag] for x in self.rec_unrotated*self.ang + self.pos]), plt="~plot", color = "b",both = True)
        self.plot_rect(np.asarray([[x.real, x.imag] for x in self.base_unrotated*self.baseAng + self.pos]), plt="~plot", color = "g", both = True)
        # print("----")
        # print(cEst)
        # print(self.pos)
        coordinates = np.asarray([[x.pos.real,x.pos.imag] for x in self.pf.particles])
        xy1 = np.c_[coordinates, np.ones_like(coordinates[:,1])]
        xy1 = np.dot(xy1,self.REF_TO_CAMERA)
        rec_centered_camera_coordinates  = (xy1[:,:2]/xy1[:,[2]])
        x = [int(x) for x in rec_centered_camera_coordinates[:,0]]
        y = [int(y) for y in rec_centered_camera_coordinates[:,1]]
        weights = [float(particle.weight) for particle in self.pf.particles]

        max = np.max(weights)
        min = np.min(weights)

        if(max != min):
            for i in range(0, len(weights)):
                weights[i] = translate(weights[i],min, max, 0.0, 1.0)

        self.visRobot("~scatter", x, y, c=weights, cmap='gray')
        self.visArena("~scatter", x, y, c=weights, cmap='gray')