from DiceTowerVision.STservo_sdk import *
import cv2 as cv
import os
import numpy as np
from DiceTowerVision.DiceTowerTools import *

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
        
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    


class DiceTower:

    # Default setting
    BAUDRATE                        = 1000000           # STServo default baudrate : 1000000
    ARM_ID                          = 1
    ARM_LOWERED_POSITION            = 0  +970          # For receiving dice
    ARM_RAISED_POSITION             = 512+970+50          # for pouring dice
    ARM_RAISING_SPEED               = 1500              # STServo moving speed
    ARM_RAISING_ACC                 = 200               # STServo moving acc
    ARM_LOWERING_SPEED              = 5000              # STServo moving speed
    ARM_LOWERING_ACC                = 100               # STServo moving acc
    ARM_RESET_SPEED                 = 200               # STServo moving speed
    ARM_RESET_ACC                   = 10                # STServo moving acc
    POSITION_TOLERANCE              = 50
    CIRCULAR_RESOLUTION             = 4096
    FLAP_ID                         = 2
    FLAP_CLOSED_POSITION            = 2048
    FLAP_OPEN_POSITION              = 1422
    FLAP_SPEED                      = 1000
    FLAP_ACC                        = 100
    FOCAL_LENGTH                    = 8.2               # mm
    PIXEL_SIZE                      = 0.00141           #mm / px (w & h)

    def __init__(self, servo_port_name, camera_id=0):
        self.camera_id = camera_id
        self.servo_port_name = servo_port_name
        self.servo_port = PortHandler(servo_port_name)
        self.servo_motor = sts(self.servo_port)
        self.connected = False
        self.camera = None
        self.camera_intrinsic = np.eye(3)

    def connect(self):
        if self.connected:
            raise Exception("DiceTower must NOT yet be connected")
        if not self.servo_port.openPort():
            raise Exception("Unable to open serial port for communication to servo: %s"%(self.servo_port_name))
        if not self.servo_port.setBaudRate(DiceTower.BAUDRATE):
            raise Exception("Unable to set serial port baudrate to %u for communication to servo"%(DiceTower.BAUDRATE))
        cam = cv.VideoCapture(self.camera_id)
        success, image = cam.read()
        if not success:
            raise Exception("Unable to capture a frame from the camera with id %u"%self.camera_id)
        self.camera = cam
        self.camera_intrinsic = get_intrinsic_camera_matrix_from_image(image, self.FOCAL_LENGTH, self.PIXEL_SIZE)
        self.connected = True
        current_position, _ = self.__get_position_speed(DiceTower.ARM_ID)
        if current_position > (DiceTower.ARM_RAISED_POSITION + DiceTower.POSITION_TOLERANCE) or current_position < (DiceTower.ARM_LOWERED_POSITION - DiceTower.POSITION_TOLERANCE) :
            raise Exception("Position out of bounds: %u"%current_position)
            

    def disconnect(self):
        self.connected = False
        self.servo_port.closePort()
        self.camera = None



    @staticmethod
    def __position_in_tolerance(measured_position, target_position, tolerance = 0):
        return abs(measured_position-target_position) <= tolerance or abs(measured_position-DiceTower.CIRCULAR_RESOLUTION-target_position) <= tolerance

    def __get_position_speed(self, id):
        if not self.connected:
            raise Exception("Servo must be connected first")
        position, speed, sts_comm_result, sts_error = self.servo_motor.ReadPosSpeed(id)
        if sts_comm_result != COMM_SUCCESS:
            self.disconnect()
            raise Exception("Communication error while getting servo position/speed: %s" % self.servo_motor.getTxRxResult(sts_comm_result))
        if sts_error != 0:
            raise Exception("Error while getting servo position/speed: %s" % self.servo_motor.getRxPacketError(sts_error))
        return (position,speed)

    def __start_move_to_position(self, id, position, speed, acc):
        if not self.connected:
            raise Exception("Servo must be connected first")
        sts_comm_result, sts_error = self.servo_motor.WritePosEx(id, position, speed, acc)
        if sts_comm_result != COMM_SUCCESS:
            self.disconnect()
            raise Exception("Communication error while moving servo: %s" % self.servo_motor.getTxRxResult(sts_comm_result))
        if sts_error != 0:
            raise Exception("Error while moving servo: %s" % self.servo_motor.getRxPacketError(sts_error))
        
    def __move_to_position_blocking(self, id, position, speed, acc, timeout=0.0):
        if not self.connected:
            raise Exception("Servo must be connected first")
        current_position, current_speed = self.__get_position_speed(id)
        if not self.__position_in_tolerance(current_position,position, DiceTower.POSITION_TOLERANCE) and current_speed == 0 :
            self.__start_move_to_position(id,position,speed,acc)
        start_time = time.monotonic()
        while not (DiceTower.__position_in_tolerance(current_position,position, DiceTower.POSITION_TOLERANCE) and current_speed == 0):
            time.sleep(0.01)
            current_position, current_speed = self.__get_position_speed(id)
            print("Position: %f ; Speed: %f"%(current_position,current_speed))
            if timeout > 0 and (time.monotonic() - start_time > timeout):
                print("Target Position: %u ; Current Position: %u ; Current Speed: %u"%(position,current_position, current_speed))
                raise TimeoutError()
        

    def __raise_arm(self, timeout=0):
        self.__move_to_position_blocking(DiceTower.ARM_ID, DiceTower.ARM_RAISED_POSITION, DiceTower.ARM_RAISING_SPEED, DiceTower.ARM_RAISING_ACC, timeout)

    def __lower_arm(self, timeout=0):
        self.__move_to_position_blocking(DiceTower.ARM_ID, DiceTower.ARM_LOWERED_POSITION, DiceTower.ARM_LOWERING_SPEED, DiceTower.ARM_LOWERING_ACC, timeout)

    def reset_arm(self, timeout=0):
        self.__move_to_position_blocking(DiceTower.ARM_ID, DiceTower.ARM_LOWERED_POSITION, DiceTower.ARM_RESET_SPEED, DiceTower.ARM_RESET_ACC, timeout)

    def __open_flap(self, timeout=0):
        self.__move_to_position_blocking(DiceTower.FLAP_ID, DiceTower.FLAP_OPEN_POSITION, DiceTower.FLAP_SPEED, DiceTower.FLAP_ACC, timeout)
    def __close_flap(self, timeout=0):
        self.__move_to_position_blocking(DiceTower.FLAP_ID, DiceTower.FLAP_CLOSED_POSITION, DiceTower.FLAP_SPEED, DiceTower.FLAP_ACC, timeout)

    def cycle_arm(self, timeout=0):
        start_time = time.monotonic()
        self.__close_flap(timeout)

        self.__raise_arm(timeout - (time.monotonic() - start_time))
        time.sleep(1.0)
        self.__lower_arm(timeout - (time.monotonic() - start_time))
        self.__open_flap(timeout - (time.monotonic() - start_time))
        time.sleep(1.0)
        self.__close_flap(timeout - (time.monotonic() - start_time))

    def get_image(self):
        if not self.connected or self.camera is None:
            raise Exception("Camera must be connected first")
        _, image = self.camera.read() #probably BGR
        return image


