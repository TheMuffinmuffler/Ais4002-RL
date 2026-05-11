import time
import numpy as np
from QUBE import QUBE
from control import COM_PORT

def check_polarity():
    port = COM_PORT
    baudrate = 115200
    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to Qube on {port}")
    except Exception as e:
        print(f"Could not connect: {e}")
        return

    print("\n--- POLARITY DIAGNOSTIC ---")
    print("1. Manually move the ARM (the horizontal part) to the RIGHT (Clockwise).")
    print("2. Check the 'Motor Angle' below. It SHOULD BE DECREASING (becoming more negative).")
    
    for _ in range(50):
        qube.update()
        print(f"Motor Angle: {qube.getMotorAngle():.1f}   ", end='\r')
        time.sleep(0.1)
    
    print("\n\n3. Manually move the PENDULUM to the RIGHT (Clockwise).")
    print("4. Check the 'Pendulum Angle' below. It SHOULD BE DECREASING.")
    
    for _ in range(50):
        qube.update()
        print(f"Pendulum Angle: {qube.getPendulumAngle():.1f}   ", end='\r')
        time.sleep(0.1)

    print("\n\n5. Final Test: Applying a tiny Positive Voltage (+1V).")
    print("   The arm SHOULD move to the LEFT (Counter-Clockwise).")
    print("   Starting in 3 seconds... CLEAR THE AREA!")
    time.sleep(3)
    
    qube.setMotorVoltage(1.0)
    start = time.time()
    while time.time() - start < 1.0:
        qube.update()
        print(f"Angle: {qube.getMotorAngle():.1f}   ", end='\r')
        time.sleep(0.02)
    
    qube.setMotorVoltage(0)
    qube.update()
    qube.master.close()
    print("\n\nTest Finished. Compare your results to the expected behavior above.")

if __name__ == "__main__":
    check_polarity()
