import time
import numpy as np
from QUBE import QUBE
from control import COM_PORT

def test_polarity():
    try:
        qube = QUBE(COM_PORT, 115200)
        print(f"Connected to Qube on {COM_PORT}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("\n--- POLARITY TEST MODE ---")
    print("This will apply a small +2V voltage for 1 second.")
    print("SAFETY: Keep hands clear of the arm!")
    input("Press Enter to start...")

    qube.resetMotorEncoder()
    qube.update()
    
    start_th = qube.getMotorAngle()
    
    # Apply +2V
    qube.setMotorVoltage(2.0)
    for _ in range(50):
        qube.update()
        time.sleep(0.02)
    
    end_th = qube.getMotorAngle()
    qube.setMotorVoltage(0.0)
    
    diff = end_th - start_th
    print(f"\nResults:")
    print(f"Start Theta: {start_th:.2f}")
    print(f"End Theta:   {end_th:.2f}")
    print(f"Change:      {diff:.2f}")
    
    if diff > 1.0:
        print("\nPOSITIVE VOLTAGE -> POSITIVE MOVEMENT (Correct)")
    elif diff < -1.0:
        print("\nPOSITIVE VOLTAGE -> NEGATIVE MOVEMENT (Inverted!)")
        print("ACTION REQUIRED: Set MOTOR_INVERT = -1.0 in config.py")
    else:
        print("\nNo significant movement detected. Check motor power.")

    if hasattr(qube, 'master'):
        qube.master.close()

if __name__ == "__main__":
    test_polarity()
