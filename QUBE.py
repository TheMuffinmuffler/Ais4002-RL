import serial
import time

RESET_ENCODER_0 = 64
RESET_ENCODER_1 = 32
SET_LED_RED = 16
SET_LED_GREEN = 8
SET_LED_BLUE = 4
SET_MOTOR_SPEED = 3


def constrain(val, _min, _max):
    return min(max(val, _min), _max)


class QUBE:
    def __init__(self, port, baudrate):
        self.master = serial.Serial(
            port=port, baudrate=baudrate, timeout=0.001, bytesize=serial.EIGHTBITS
        )
        self.master.reset_input_buffer()
        self.master.reset_output_buffer()
        
        self.rpm = 0
        self.voltage = 0
        self.current = 0
        self.motorAngle = 0
        self.pendulumAngle = 0
        
        # Initialize output with safe defaults
        # 0: Reset Enc 0
        # 1: Reset Enc 1
        # 2-3: Red MSB/LSB
        # 4-5: Green MSB/LSB
        # 6-7: Blue MSB/LSB
        # 8-9: Motor MSB/LSB (999 is 0V)
        self.output = [0] * 10
        self.setMotorSpeed(0) # Sets 8-9 to 999
        self.setRGB(0, 0, 0)
        
        self.startTime = time.time()

    # Getters
    def getMotorAngle(self):
        return self.motorAngle

    def getPendulumAngle(self):
        return self.pendulumAngle

    def getMotorRPM(self):
        return self.rpm

    def getMotorCurrent(self):
        return self.current

    # Setters
    def resetMotorEncoder(self):
        self.output[0] = 1

    def resetPendulumEncoder(self):
        self.output[1] = 1

    def setMotorSpeed(self, speed):
        # speed is -999 to 999
        speed = constrain(speed, -999, 999)
        val = int(speed + 999)
        self.output[8] = (val >> 8) & 0xFF
        self.output[9] = val & 0xFF

    def setMotorVoltage(self, volts):
        self.voltage = min(24, max(-24, volts))
        speed = (volts / 24.0) * 999
        self.setMotorSpeed(speed)

    def setRGB(self, r, g, b):
        r = int(constrain(r, 0, 999))
        g = int(constrain(g, 0, 999))
        b = int(constrain(b, 0, 999))
        
        self.output[2] = (r >> 8) & 0xFF
        self.output[3] = r & 0xFF
        self.output[4] = (g >> 8) & 0xFF
        self.output[5] = g & 0xFF
        self.output[6] = (b >> 8) & 0xFF
        self.output[7] = b & 0xFF

    def decodeEncoderAngle(self, data):
        rev_MSB = data[0]
        rev_LSB = data[1]
        ang_MSB = data[2]
        ang_LSB = data[3]

        dir = rev_MSB >> 7
        revolutions = (rev_LSB) + ((rev_MSB & 0x7F) << 8)
        
        # If MSB is set, it's negative
        if dir:
            revolutions = -revolutions

        angleInt = ang_MSB * 2 + (ang_LSB >> 7)
        angleDec = (ang_LSB & 0b01111111) * 0.01
        angle = angleInt + angleDec
        
        if dir:
            angle = -angle

        return revolutions * 360.0 + angle

    def decodeMotorRPM(self, data):
        rpm_MSB = data[0]
        rpm_LSB = data[1]
        dir = rpm_MSB >> 7
        rpm = ((rpm_MSB & 0x7F) << 8) | rpm_LSB
        if dir:
            rpm = -rpm
        return rpm

    def decodeMotorCurrent(self, data):
        current_MSB = data[0]
        current_LSB = data[1]
        current = (current_MSB << 8) | current_LSB
        return current

    def update(self):
        # 1. Send 10 bytes of command
        self.master.write(bytearray(self.output))
        
        # Clear reset flags after sending
        self.output[0] = 0
        self.output[1] = 0
        
        # 2. Robust Read: Flush old data and wait for a FRESH packet.
        # This eliminates latency-induced instability.
        self.master.reset_input_buffer()
        
        # Wait for exactly 12 bytes (timeout after 50ms)
        start_wait = time.time()
        while self.master.in_waiting < 12:
            if time.time() - start_wait > 0.05:
                return # Skip this update if hardware is slow
            
        raw_data = self.master.read(12)
        if len(raw_data) == 12:
            new_motor_angle = self.decodeEncoderAngle(raw_data[0:4])
            new_pendulum_angle = self.decodeEncoderAngle(raw_data[4:8])
            
            # 3. Jump Protection: Ignore garbage data (e.g. >180 deg jump in 20ms)
            if abs(new_motor_angle - self.motorAngle) < 180:
                self.motorAngle = new_motor_angle
            
            if abs(new_pendulum_angle - self.pendulumAngle) < 180:
                self.pendulumAngle = new_pendulum_angle
                
            self.rpm = self.decodeMotorRPM(raw_data[8:10])
            self.current = self.decodeMotorCurrent(raw_data[10:12])
