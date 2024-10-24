import pyxinput
import time

controller = pyxinput.vController() # use into the vController constr the percent=False if you want to use exact values
def send_xbox_input(steering, throttle, brake):

    controller.set_value('AxisLx', int(steering)) 
    controller.set_value('TriggerR', int(throttle)) 
    controller.set_value('TriggerL', int(brake)) 
    controller.set_value('AxisLy', 0)
    controller.set_value('AxisRx', 0)
    controller.set_value('AxisRy', 0)
    controller.set_value('Dpad', 8)
    controller.set_value('BtnA', 0)
    controller.set_value('BtnB', 0)
    controller.set_value('BtnX', 0)
    controller.set_value('BtnY', 0)

    print(f"Steering: {steering}, Throttle: {throttle}, Brake: {brake}")

while True:
    send_xbox_input(steering=0.5, throttle=1.0, brake=0.0)  
    time.sleep(2)

    send_xbox_input(steering=-0.5, throttle=0.0, brake=1.0)  
    time.sleep(2)
