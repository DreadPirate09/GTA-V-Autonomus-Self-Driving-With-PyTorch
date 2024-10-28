import pyxinput
import time

class Pilot(object):

    def __init__(self,name='Bob',percent=True):
        self.name = name
        self.controller = pyxinput.vController(percent=True)

    def sendIt(self, steering, throttle, brake):
        self.controller.set_value('AxisLx', steering - 0.5) 
        self.controller.set_value('TriggerR', throttle) 
        self.controller.set_value('TriggerL', brake) 
        print(f"Steering: {steering}, Throttle: {throttle}, Brake: {brake}")