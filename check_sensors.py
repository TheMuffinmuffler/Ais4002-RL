import time
import numpy as np
from QUBE import QUBE
from control import COM_PORT

def check_sensors():
    port = COM_PORT
    baudrate = 115200
    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to Qube on {port}")
    except Exception as e:
        print(f"Could not connect to {port}: {e}")
        return

    print("\n--- SENSOR DIAGNOSTIC ---")
    print("1. Hold the pendulum HANGING DOWN and STILL.")
    print("2. Reseting encoders in 3 seconds...")
    time.sleep(3)
    
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.setMotorVoltage(0)
    qube.update()
    
    print("\nMove the hardware manually and check the values below.")
    print("Press Ctrl+C to stop.\n")
    print(f"{'Motor (Theta)':>15} | {'Pendulum (Alpha)':>15} | {'Current (mA)':>12}")
    print("-" * 50)

    try:
        while True:
            qube.update()
            theta = qube.getMotorAngle()
            alpha = qube.getPendulumAngle()
            current = qube.getMotorCurrent()
            
            # Print on one line, updating
            print(f"{theta:15.2f} | {alpha:15.2f} | {current:12.1f}", end='\r')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nDiagnostic Stopped.")
    finally:
        qube.setMotorVoltage(0)
        qube.master.close()

if __name__ == "__main__":
    check_sensors()
