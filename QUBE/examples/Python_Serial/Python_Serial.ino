#include "QUBE.hpp"

QUBE qube;

void setup() {
  Serial.begin(115200);
  qube.begin();
  qube.resetMotorEncoder();
  qube.resetPendulumEncoder();
  qube.setMotorVoltage(0);
  qube.setRGB(0, 999, 999);
  qube.update();
}

void receiveData() {
  // Wait for 10 bytes of command
  if (Serial.available() >= 10) {    
    bool resetMotorEncoder = Serial.read();
    bool resetPendulumEncoder = Serial.read();

    int r_MSB = Serial.read();
    int r_LSB = Serial.read();
    int r = (r_MSB << 8) | r_LSB;

    int g_MSB = Serial.read();
    int g_LSB = Serial.read();
    int g = (g_MSB << 8) | g_LSB;

    int b_MSB = Serial.read();
    int b_LSB = Serial.read();
    int b = (b_MSB << 8) | b_LSB;

    int motorCommand_MSB = Serial.read();
    int motorCommand_LSB = Serial.read();
    int motorCommand = (motorCommand_MSB << 8) | motorCommand_LSB;
    motorCommand -= 999;

    if (resetMotorEncoder) {
      qube.resetMotorEncoder();
    }
    if (resetPendulumEncoder) {
      qube.resetPendulumEncoder();
    }
    qube.setRGB(r, g, b);
    qube.setMotorSpeed(motorCommand);
    
    // Send 12 bytes of response immediately after receiving
    sendData();
  }
}

void sendEncoderData(bool isPendulum) {
  float encoderAngle = isPendulum ? qube.getPendulumAngle(false) : qube.getMotorAngle(false);
  
  long revolutions = (long)(encoderAngle / 360.0);
  float _angle = encoderAngle - (revolutions * 360.0);
  
  bool negative = false;
  if (encoderAngle < 0) {
    negative = true;
    revolutions = abs(revolutions);
    _angle = abs(_angle);
  }

  long angleInt = (long)_angle;
  long angleDecimal = (long)((_angle - angleInt) * 100);

  // Pack revolutions: MSB = Sign, Bits 14-0 = Count
  uint16_t revPack = (uint16_t)revolutions & 0x7FFF;
  if (negative) revPack |= 0x8000;

  // Pack angle: Bits 15-7 = Int (0-360), Bits 6-0 = Dec (0-100)
  uint16_t angPack = ((uint16_t)angleInt << 7) | ((uint16_t)angleDecimal & 0x7F);

  Serial.write((byte)(revPack >> 8));
  Serial.write((byte)(revPack & 0xFF));
  Serial.write((byte)(angPack >> 8));
  Serial.write((byte)(angPack & 0xFF));
}

void sendRPMData() {
  long rpm = (long)qube.getRPM();
  bool negative = rpm < 0;
  uint16_t rpmPack = (uint16_t)abs(rpm) & 0x7FFF;
  if (negative) rpmPack |= 0x8000;

  Serial.write((byte)(rpmPack >> 8));
  Serial.write((byte)(rpmPack & 0xFF));
}

void sendCurrentData() {
  uint16_t current = (uint16_t)abs((long)qube.getMotorCurrent());
  Serial.write((byte)(current >> 8));
  Serial.write((byte)(current & 0xFF));
}

void sendData() {
  sendEncoderData(false); // Motor
  sendEncoderData(true);  // Pendulum
  sendRPMData();
  sendCurrentData();
}

void loop() {
  qube.update();
  receiveData();
}
