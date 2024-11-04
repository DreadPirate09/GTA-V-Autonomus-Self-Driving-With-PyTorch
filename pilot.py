import pyxinput
import time

class Pilot(object):

    def __init__(self,name='Bob',percent=True,max_throttle=1.0):
        self.name = name
        self.controller = pyxinput.vController(percent=True)
        self.max_throttle = max_throttle

    def sendIt(self, steering, throttle, brake, speed):
        steering = steering*2.0 - 1.0
        if speed < 30 and throttle < 0.65:
            throttle = 0.85
        if speed > 70:
            throttle = 0.0
        # if throttle > self.max_throttle:
        #     throttle = self.max_throttle

        steering = steering * 1.6
        self.controller.set_value('AxisLx', steering) 
        self.controller.set_value('TriggerR', throttle) 
        self.controller.set_value('TriggerL', brake) 

        print(f"Steering: {steering}, Throttle: {throttle}, Brake: {brake}")