import pyxinput
import time

class Pilot(object):

    def __init__(self,name='Bob',percent=True):
        self.name = name
        self.controller = pyxinput.vController(percent=True)

    def sendIt(self, steering, throttle, brake):
        self.controller.set_value('AxisLx', (steering*2.0 - 1.0) * 1.3) 
        self.controller.set_value('TriggerR', throttle) 
        self.controller.set_value('TriggerL', brake) 
        print(f"Steering: {(steering*2.0 - 1 ) * 1.3}, Throttle: {throttle}, Brake: {brake}")