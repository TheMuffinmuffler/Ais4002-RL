import time
from QUBE import QUBE
from control import COM_PORT

def test_motor():
    port = COM_PORT
    baudrate = 115200
    try:
        qube = QUBE(port, baudrate)
        print(f"Connected to Qube on {port}")
    except Exception as e:
        print(f"Could not connect to {port}: {e}")
        return

    print("\n--- MOTOR POLARITY TEST ---")
    print("1. Clear the area around the Qube arm.")
    print("2. I will apply +2V for 1 second, then -2V for 1 second.")
    print("3. Starting in 3 seconds...")
    time.sleep(3)

    try:
        print("Applying +2V (Should move CCW/Right)...")
        qube.setMotorVoltage(2.0)
        start = time.time()
        while time.time() - start < 1.0:
            qube.update()
            print(f"Angle: {qube.getMotorAngle():.2f}", end='\r')
            time.sleep(0.02)
        
        qube.setMotorVoltage(0)
        qube.update()
        time.sleep(0.5)

        print("\nApplying -2V (Should move CW/Left)...")
        qube.setMotorVoltage(-2.0)
        start = time.time()
        while time.time() - start < 1.0:
            qube.update()
            print(f"Angle: {qube.getMotorAngle():.2f}", end='\r')
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        qube.setMotorVoltage(0)
        qube.update()
        qube.master.close()
        print("\nTest Finished.")

if __name__ == "__main__":
    test_motor()
