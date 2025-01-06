import pyxinput
import time
import pandas as pd

TRAIN_DATA_FILE = "data/data.csv"

class Pilot(object):

    def __init__(self,name='Bob',percent=True):
        self.name = name
        self.controller = pyxinput.vController(percent=True)
        self.avg_speed = self.getAvgSpeed()

    def sendIt(self, steering, throttle, brake, speed):
        steering = steering*2.0 - 1.0
        if throttle < 0.25 and abs(steering) < 0.3:
            throttle = 0.30
        if speed < 45 and throttle < 0.65 and abs(steering) < 0.3:
            throttle = 0.75
        if speed > self.avg_speed:
            throttle = throttle / 3.0

        steering = steering * 1.9
        self.controller.set_value('AxisLx', steering) 
        self.controller.set_value('TriggerR', throttle) 
        self.controller.set_value('TriggerL', brake) 

        print(f"steering wheel angle: {steering:.4f}, power: {throttle:.4f}, km/h: {speed}")

    def resetController(self):
        self.controller.set_value('AxisLx', 0.0)
        self.controller.set_value('TriggerR', 0.0) 
        self.controller.set_value('TriggerL', 0.0) 

    def pullHandBreak(self):
        self.controller.set_value('TriggerL', 1.0)
        time.sleep(0.5)
        self.controller.set_value('TriggerL',0.0)

    def releaseHandBreak(self):
        self.controller.set_value('TriggerL', 0.0)

    def getAvgSpeed(self):
        data = pd.read_csv(TRAIN_DATA_FILE)
        avg_speed = sum([data.iloc[x,3] for x in range(len(data))])/len(data)

        return avg_speed
