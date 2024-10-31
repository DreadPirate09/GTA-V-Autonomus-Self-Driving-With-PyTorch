import pyxinput
import time

class Pilot(object):

    def __init__(self,name='Bob',percent=True,max_throttle=1.0):
        self.name = name
        self.controller = pyxinput.vController(percent=True)
        self.max_throttle = max_throttle

    def sendIt(self, steering, throttle, brake):
        steering = steering*2.0 - 1.0
        if throttle > self.max_throttle:
            throttle = self.max_throttle
        self.controller.set_value('AxisLx', steering * 1.5) 
        self.controller.set_value('TriggerR', throttle) 
        self.controller.set_value('TriggerL', brake) 

        print(f"Steering: {steering}, Throttle: {throttle}, Brake: {brake}")