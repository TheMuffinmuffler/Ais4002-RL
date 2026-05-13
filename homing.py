import time
import numpy as np

def auto_home(qube):
    """
    Automatically finds the center of the motor arm by touching 
    the physical hard-stops. Sets the center to 0.
    """
    print("\n--- AUTO-HOMING ROUTINE ---")
    print("Please keep hands clear. Finding arm center...")
    qube.setRGB(0, 0, 999) # Blue for homing

    def find_limit(voltage_sign):
        target_v = 1.2 * voltage_sign # Increased to 1.2V to overcome mid-travel friction
        v_current = 0.0
        ramp = 0.04 # Slightly faster ramp
        
        stable_count = 0
        prev_angle = qube.getMotorAngle()
        
        # Ramp up voltage and wait for stop
        for i in range(500): 
            if abs(v_current) < abs(target_v):
                v_current += ramp * voltage_sign
            
            qube.setMotorVoltage(v_current)
            qube.update()
            
            curr_angle = qube.getMotorAngle()
            curr_rpm = qube.getMotorRPM()
            
            # Stall Detection:
            # 1. Position hasn't changed much
            # 2. RPM is very low
            # 3. Voltage is high enough to be moving
            if abs(curr_angle - prev_angle) < 0.05 and abs(curr_rpm) < 20 and abs(v_current) > 0.8:
                stable_count += 1
            else:
                stable_count = 0
            
            if stable_count > 20: # Must be stalled for 20 frames (~1.0s)
                print(f"  Limit detected at {curr_angle:.1f} deg.")
                break
                
            prev_angle = curr_angle
            time.sleep(0.05)
        
        limit_pos = qube.getMotorAngle()
        # Back off slowly
        qube.setMotorVoltage(-target_v * 0.4)
        time.sleep(0.8)
        qube.setMotorVoltage(0)
        return limit_pos

    # 1. Find Left Limit
    print("Searching for LEFT limit... (Wait for stall)")
    left_limit = find_limit(1.0) 
    
    time.sleep(1.0)

    # 2. Find Right Limit
    print("Searching for RIGHT limit... (Wait for stall)")
    right_limit = find_limit(-1.0) 

    # 3. Calculate and Move to Center
    center = (left_limit + right_limit) / 2.0
    print(f"Calculated Center: {center:.1f}. Moving to center (very slowly)...")
    
    # Proportional move to center (capped very low for safety)
    for _ in range(400): # Increased timeout for slow move
        qube.update()
        err = center - qube.getMotorAngle()
        if abs(err) < 0.2:
            break
        v = np.clip(0.03 * err, -0.4, 0.4) # Capped at 0.4V for ultra-slow move
        qube.setMotorVoltage(v)
        time.sleep(0.02)
    
    qube.setMotorVoltage(0)
    time.sleep(0.5)
    
    # 4. Final Zeroing
    qube.resetMotorEncoder()
    qube.update()
    print("--- HOMING COMPLETE: Arm is now centered at 0.0 ---")
    qube.setRGB(0, 999, 0) # Green for success
    time.sleep(1.0)
