import time
import numpy as np
from QUBE import QUBE
from control import COM_PORT
from config import THETA_INVERT, PENDULUM_INVERT, MOTOR_INVERT

def live_test():
    try:
        qube = QUBE(COM_PORT, 115200)
        print(f"Connected to QUBE on {COM_PORT}")
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        qube.setRGB(0, 999, 0) # Green for test
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("\n" + "="*50)
    print("LIVE TELEMETRY TEST (30 SECONDS)")
    print("="*50)
    print("1. ARM: Move it RIGHT (Clockwise). Model Angle should DECREASE.")
    print("2. PENDULUM: Move it RIGHT (Clockwise). Model Angle should DECREASE.")
    print("3. VOLTAGE: I will not apply voltage yet. Just check manual movement.")
    print("="*50)
    print(f"{'Time':>6} | {'Motor (Raw)':>12} | {'Motor (Model)':>12} | {'Pend (Raw)':>12} | {'Pend (Model)':>12}")
    print("-" * 65)

    start_time = time.time()
    try:
        while time.time() - start_time < 30:
            qube.update()
            
            raw_motor = qube.getMotorAngle()
            raw_pend = qube.getPendulumAngle()
            
            # This is how deploy_common.py sees them
            model_motor = raw_motor * THETA_INVERT
            model_pend = (raw_pend * PENDULUM_INVERT + 180.0) % 360.0 - 180.0
            
            elapsed = time.time() - start_time
            print(f"{elapsed:6.1f} | {raw_motor:12.1f} | {model_motor:12.1f} | {raw_pend:12.1f} | {model_pend:12.1f}", end='\r')
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        qube.setMotorVoltage(0)
        qube.setRGB(999, 0, 0)
        qube.master.close()
        print("\n" + "="*50)
        print("Test Finished.")

if __name__ == "__main__":
    live_test()
