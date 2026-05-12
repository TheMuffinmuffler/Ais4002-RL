import time
import numpy as np
from QUBE import QUBE
from control import COM_PORT

def check():
    try:
        qube = QUBE(COM_PORT, 115200)
        print(f"Connected to Qube on {COM_PORT}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Calibrating encoders... (Pendulum should be HANGING DOWN)")
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    time.sleep(1)

    print("\n--- SENSOR CHECK MODE ---")
    print("Move the arm and pendulum manually to verify mapping:")
    print("1. Rotate ARM LEFT (clockwise from top) -> Theta should INCREASE.")
    print("2. Lift PENDULUM to TOP -> Alpha should go toward 0.")
    print("3. Pendulum at BOTTOM -> Alpha should be near -180 or 180.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            qube.update()
            
            # Raw values from QUBE.py
            th_raw = qube.getMotorAngle()
            al_raw = qube.getPendulumAngle()
            
            # Mapping Logic (Matches what we will use in Deployment)
            # 1. Wrap to [0, 360)
            al_wrapped = al_raw % 360
            # 2. Shift so BOTTOM is -180/180 and TOP is 0
            # Since reset at BOTTOM=0, then 0-180 = -180.
            alpha_deg = al_wrapped - 180
            if alpha_deg > 180: alpha_deg -= 360
            if alpha_deg < -180: alpha_deg += 360

            theta_deg = th_raw # Arm usually doesn't need wrapping unless it spins many times

            print(f"RAW: Th:{th_raw:7.1f} Al:{al_raw:7.1f} | MAPPED: Theta:{theta_deg:6.1f} Alpha:{alpha_deg:6.1f}", end='\r')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting Sensor Check.")
    finally:
        if hasattr(qube, 'master'):
            qube.master.close()

if __name__ == "__main__":
    check()
