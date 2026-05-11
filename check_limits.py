import time
import numpy as np
from QUBE import QUBE
from control import COM_PORT

def check_limits():
    # Initialize connection
    port = COM_PORT
    baudrate = 115200
    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to Qube on {port}")
    except Exception as e:
        print(f"Could not connect to {port}: {e}")
        return

    print("\n--- PHYSICAL LIMIT TEST ---")
    print("1. Set the arm to the CENTER (facing you).")
    print("2. I will zero the encoders in 3 seconds...")
    time.sleep(3)
    
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.setRGB(999, 999, 999) # White
    print("ENCODERS ZEROED.")
    print("\nMANUALLY MOVE THE ARM to find your safe limits.")
    print("Watch the 'Theta' value below.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            qube.update()
            theta_deg = qube.getMotorAngle()
            alpha_deg = qube.getPendulumAngle()
            theta_rad = np.deg2rad(theta_deg)
            
            # Change LED color based on a "suggested" 90-degree limit
            if abs(theta_deg) > 90:
                qube.setRGB(999, 0, 0) # Red (Warning)
            else:
                qube.setRGB(0, 999, 0) # Green (Safe)

            print(f"Theta: {theta_deg:6.1f} deg ({theta_rad:5.2f} rad) | Alpha: {alpha_deg:6.1f} deg", end='\r')
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        qube.setRGB(0, 0, 0)
        if hasattr(qube, 'master'):
            qube.master.close()

if __name__ == "__main__":
    check_limits()
