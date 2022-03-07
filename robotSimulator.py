# !/usr/bin/python3
# file robotSimulator.py simulates a robot in an arena

import sys, os
if 'pyckbot/hrb/' not in sys.path:
    sys.path.append(os.path.expanduser('~/pyckbot/hrb/'))

from movePlans import *
from sensorPlanTCP import SensorPlanTCP
from joy import JoyApp, progress
from joy.decl import *
from joy.plans import Plan
from waypointShared import (
  WAYPOINT_HOST, WAYPOINT_MSG_PORT, APRIL_DATA_PORT, Sensor
  )
from socket import (
  socket, AF_INET,SOCK_DGRAM, IPPROTO_UDP, error as SocketError,
  )
from pylab import randn,dot,mean,exp,newaxis
import math

# Added or modified
if '-r' in sys.argv:
    print('Using real robot')
    from myRobotRealIX import RobotSim, RobotSimInterface
else:
    from myRobotSimIX import RobotSim, RobotSimInterface

class RobotSimulatorApp( JoyApp ):
  """Concrete class RobotSimulatorApp <<singleton>>
     A JoyApp which runs the DummyRobotSim robot model in simulation, and
     emits regular simulated tagStreamer message to the desired waypoint host.

     Used in conjection with waypointServer.py to provide a complete simulation
     environment for Project 1
  """
  def __init__(self,wphAddr=WAYPOINT_HOST,wphPort=WAYPOINT_MSG_PORT,*arg,**kw):
      """
      Initialize the simulator
      """
      JoyApp.__init__( self,
        confPath="$/cfg/JoyApp.yml", *arg, **kw
        )
      self.srvAddr = (wphAddr, wphPort)
      # ADD pre-startup initialization here, if you need it

  def onStart( self ):
    """
    Sets up the JoyApp and configures the simulation
    """
    ### DO NOT MODIFY ------------------------------------------
    # Set up socket for emitting fake tag messages
    s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
    s.bind(("",0))
    self.sock = s
    # Set up the sensor receiver plan
    self.sensor = SensorPlanTCP(self,server=self.srvAddr[0],port=self.srvAddr[1])
    self.sensor.start()
    self.timeForStatus = self.onceEvery(1)
    self.timeForLaser = self.onceEvery(1/15.0)
    self.timeForFrame = self.onceEvery(1/20.0)
    progress("Using %s:%d as the waypoint host" % self.srvAddr)
    self.T0 = self.now
    ### MODIFY FROM HERE ------------------------------------------
    self.robSim = RobotSim(fn=None, app=self)
    self.move = MoveDistClass(self, self.robSim)
    self.liftWheels = LiftWheelsClass(self, self.robSim)
    self.turn = TurnClass(self, self.robSim)
    self.autoP = Auto(self, self.robSim, self.sensor)

  def showSensors( self ):
    """
    Display sensor readings
    """
    # This code should help you understand how you access sensor information
    ts,f,b = self.sensor.lastSensor
    if ts:
      progress( "Sensor: %4d f %d b %d" % (ts-self.T0,f,b)  )
    else:
      progress( "Sensor: << no reading >>" )
    ts,w = self.sensor.lastWaypoints
    if ts:
      progress( "Waypoints: %4d " % (ts-self.T0) + str(w))
    else:
      progress( "Waypoints: << no reading >>" )

  def emitTagMessage( self ):
    """Generate and emit and update simulated tagStreamer message"""
    #### DO NOT MODIFY --- it WILL break the simulator
    self.robSim.refreshState()
    # Get the simulated tag message
    msg = self.robSim.getTagMsg()
    # Send message to waypointServer "as if" we were tagStreamer
    self.sock.sendto(msg.encode("ascii"), (self.srvAddr[0],APRIL_DATA_PORT))

  def stop_all_plans(self):
    try:
      self.autoP.stop()
      self.move.stop()
      self.turn.stop()
    except:
      pass

  def onEvent( self, evt ):
    #### DO NOT MODIFY --------------------------------------------
    # periodically, show the sensor reading we got from the waypointServer
    if self.timeForStatus():
      self.showSensors()
      progress( self.robSim.logLaserValue(self.now) )
      # generate simulated laser readings
    elif self.timeForLaser():
      self.robSim.logLaserValue(self.now)
    # update the robot and simulate the tagStreamer
    if self.timeForFrame():
      self.emitTagMessage()
    #### MODIFY FROM HERE ON ----------------------------------------
    key_set = [K_a,K_UP,K_DOWN,K_SPACE,K_LEFT,K_RIGHT,K_r,K_q]
    if evt.type == KEYDOWN:
      say = "(2022W-P1-GREEN) "
      ##TURN AND STEP SIZE CONSTANTS
      da, dx = 10 *(math.pi/180), 5 
      if evt.key in key_set:
        self.stop_all_plans()
      if evt.key == K_a:
        self.autoP.start()
        return progress(say + "Auto starting")
      if evt.key == K_UP:
        self.move.dist = dx
        self.move.start()
        return progress(say + "Moving Forward")
      if evt.key == K_DOWN:
        self.move.dist = -dx
        self.move.start()
        return progress(say + "Moving Back")
      if evt.key == K_SPACE:
        self.liftWheels.start()
        return progress(say + "Lifting Wheels")
      if evt.key == K_LEFT:
        self.turn.absolute = False
        self.turn.ang = da
        self.turn.start()
        return progress(say + "Turn left")
      if evt.key == K_RIGHT:
        self.turn.absolute = False
        self.turn.ang = -da
        self.turn.start()
        return progress(say + "Turn right")
      if evt.key == K_r:
        return progress(say + "RESET")
      if evt.key == K_q:
        progress("--------EDR--------")
        self.stop()
    ### DO NOT MODIFY -----------------------------------------------
      else:# Use superclass to show any other events
        return JoyApp.onEvent(self,evt)
    return # ignoring non-KEYDOWN events


if __name__=="__main__":
  from sys import argv
  print("""
  Running the robot simulator

  Listens on local port 0xBAA (2986) for incoming waypointServer
  information, and also transmits simulated tagStreamer messages to
  the waypointServer at ip and port given on commandline

  USAGE:
    %s
        Connect to default host and port
    %s -r
        Use real robot
    %s <host>
        Connect to specified host on default port
    %s <host> <port>
        Connect to specified host on specified port
  """ % ((argv[0],)*4))
  if '-r' in sys.argv:
    sys.argv.remove('-r')
    motorNames = {0x08:"wheelMotorFront",
                  0x3C:"wheelMotorBack",
                  0x14:"liftServoFront",
                  0x93:"liftServoBack",
                  0x32:"spinMotor"}
    robot = {'count':5, 'names': motorNames}
  else:
    robot = None
  cfg = {'windowSize' : [160,120]}
  if len(argv)>2:
    app=RobotSimulatorApp(wphAddr=argv[1],wphPort=int(argv[2]),robot=robot,cfg=cfg)
  elif len(argv)==2:
    app=RobotSimulatorApp(wphAddr=argv[1],robot=robot,cfg=cfg)
  else:
    app=RobotSimulatorApp(robot=robot,cfg=cfg)
  app.run()