# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 20:31:13 2014

@author: shrevzen-home
"""
import sys, os
if 'pyckbot/hrb/' not in sys.path:
    sys.path.append(os.path.expanduser('~/pyckbot/hrb/'))

from gzip import open as opengz
from json import dumps as json_dumps
from numpy import (
    asarray, asfarray, dot, c_, newaxis, mean, exp, sum,
    sqrt, any, isnan, sin, cos, pi, angle
    )
from numpy.linalg import svd
from numpy.random import randn
from waypointShared import *
import numpy as np
import math

from joy.plans import Plan
# from robotSimIX import RobotSimInterface # DO THIS INSTEAD?
class RobotSimInterface( object ):
    """
    Abstract superclass RobotSimInterface defines the output-facing interface
    of a robot simulation.

    Subclasses of this class must implement all of the methods
    """
    def __init__(self, fn=None):
        """
        INPUT:
            fn -- filename / None -- laser log name to use for logging simulated
            laser data. None logged if name is None

        ATTRIBUTES:
            tagPos -- 4x2 float array -- corners of robot tag
            laserAxis -- 2x2 float array -- two points along axis of laser
            waypoints -- dict -- maps waypoint tag numbers to 4x2 float
                arrays of the tag corners
        """
        # Do nothing
        pass

    def refreshState( self ):
        """<<pure>> refresh the value of self.tagPos and self.laserAxis"""
        print("<<< MUST IMPLEMENT THIS METHOD >>>")

    def getTagMsg( self ):
        """
        Using the current state, generate a TagStreamer message simulating
        the robot state
        """
        # Do nothing
        return ""

    def logLaserValue( self, now ):
        """
        Using the current state, generate a fictitious laser pointer reading
        INPUT:
            now -- float -- timestamp to include in the message

        OUTPUT: string of human readable message (not what is in log)
        """
        # Do nothing
        return ""

class RobotSim( RobotSimInterface ):
    def __init__(self, app=None, *args, **kw):
        RobotSimInterface.__init__(self, *args, **kw)
        # initialize motors
        self.app = app
        self.servo = self.app.robot.at
        self.liftAngle = 2500
        self.manualSpeed = 10
        self.spinMotorOffset = 4500
        self.servo.wheelMotorFront.set_mode('cont')
        self.servo.wheelMotorBack.set_mode('cont')
        self.servo.wheelMotorFront.set_speed(10)
        self.servo.wheelMotorBack.set_speed(10)
        self.servo.liftServoFront.set_mode(0)
        self.servo.liftServoBack.set_mode(0)
        self.servo.liftServoFront.set_speed(15)
        self.servo.liftServoBack.set_speed(7)
        self.servo.liftServoFront.set_pos(-self.liftAngle)
        self.servo.liftServoBack.set_pos(-self.liftAngle)
        self.servo.spinMotor.set_mode(2)
        self.servo.spinMotor.set_speed(7)

    def wheelsDown(self):
        return self.servo.liftServoFront.get_pos() < -self.liftAngle/2

    def turn(self, ang, absolute=False):
        degrees = - 1.0 * ang * 180 / pi
        if self.wheelsDown():
            yield self.liftWheels()
        currentPos = self.servo.spinMotor.get_pos()
        movePos = currentPos-degrees*100
        if absolute:
            movePos = self.spinMotorOffset-degrees*100
        self.servo.spinMotor.set_pos(movePos)
        yield

    def liftWheels(self):
        self.servo.wheelMotorFront.set_pos(self.servo.wheelMotorFront.get_pos())
        self.servo.wheelMotorBack.set_pos(self.servo.wheelMotorBack.get_pos())
        if self.wheelsDown():
            self.servo.liftServoFront.set_pos(0)
            self.servo.liftServoBack.set_pos(0)
        else:
            self.servo.liftServoFront.set_pos(-self.liftAngle)
            self.servo.liftServoBack.set_pos(-self.liftAngle)
        yield

    def move(self, dist):
        if not self.wheelsDown():
            yield self.liftWheels()
        #numRotations is postive for forward and negative for backward
        stepSize = 0.1;
        #Front motor
        posFront = self.servo.wheelMotorFront.get_pos()
        posBack = self.servo.wheelMotorBack.get_pos()

        #Original position
        posFrontOrig = posFront
        posBackOrig = posBack

        numSteps = abs(math.floor(numRotations/stepSize))
        print(numSteps)
        for i in range(1,int(numSteps+1)):
            #print(i)
            posFront += stepSize*36000*np.sign(dist)
            posBack  += stepSize*36000*np.sign(dist)*-1
            self.servo.wheelMotorFront.set_pos(posFront)
            self.servo.wheelMotorBack.set_pos(posBack)
            yield self.app.move.forDuration((stepSize / self.manualSpeed) * 60 + 0.05)
            #yield self.app.move.forDuration(1)

        finalPosFront = posFrontOrig + dist*36000
        finalPosBack  = posBackOrig  + dist*36000*-1
        self.servo.wheelMotorFront.set_pos(finalPosFront)
        self.servo.wheelMotorBack.set_pos(finalPosBack)
        yield

    def turnTagTo(self, tag_heading):
        # Turns the tag mounted on the robot to a given heading
        # tag_heading is a complex number describing an angle relative to the +x axis of the intended heading
        # The tag is intended to have one 'sensor' on each side of the line, so the intended heading is not the direction of the front tag but instead has the front on the left and back on the right
        # In the following diagram the caret symbol (^) is tag_heading, f is the front sensor, and b is the back sensor
        #   ^
        # f—o—b
        #   |
        # For example, when tag_heading == 1 the tag is pointing to +x
        # when tag_heading == -1j the tag is pointing to -y
        self.motors[-1].set_pos(angle(tag_heading)*18000/pi)

    def refreshState(self):
        pass
