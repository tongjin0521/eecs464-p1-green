# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 20:31:13 2014

@author: shrevzen-home
"""

import sys, os

if 'pyckbot/hrb/' not in sys.path:
    sys.path.append(os.path.expanduser('~/pyckbot/hrb/'))

import numpy as np
from numpy.random import randn
from waypointShared import *
import random

#returns if a is equal to b within noise margins
def about_equal(a, b, noise):
    if((a <= (b + noise)) and (a >= (b - noise))):
        return True
    else:
        return False

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
        move_dist_noise = 0.1
        move_angle_noise = np.pi / 180 * 0.1

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
        #print("updating")
        #print("f: " + str(measured_f))
        #print("b: " + str(measured_b))
        noise_est = 2.0
        ##discard distance measurements if they are about 0
        '''
        if(about_equal(measured_f, 0.0, noise_est) and about_equal(measured_b, 0.0, noise_est)):
            return
        elif(about_equal(measured_f, 0.0, noise_est)):
            real_dist = measured_b
        elif(about_equal(measured_b, 0.0, noise_est)):
            real_dist = measured_f
        else:
        '''
        real_dist = (measured_f + measured_b)/2
        
        max_sense = 0
        #for particle in self.particles:
        for i in range(0, len(self.particles)):
            #particle_distance = self.Sensor.sense(None, self.particles[i].pos, a, b) ##TODO
            particle_distance = float(self.Sensor.sense(None, a, b, self.particles[i].pos, 10.5))
            #print(particle_distance)
            if(particle_distance > max_sense):
                max_sense = particle_distance
            normalized_distance = abs(particle_distance - real_dist) / 255.0
            scale_factor = (1 - normalized_distance * 1)
            if(about_equal(real_dist, particle_distance, noise_est)):
                self.particles[i].weight *= 1
            else:
                self.particles[i].weight *= scale_factor
        self.particles = self.normalized_particles(self.particles)
        #print(max_sense)
        print("max particle: " + str(max(x.weight for x in self.particles)))

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
    