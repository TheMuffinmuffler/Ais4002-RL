import csv
import os
import time
import threading
import queue

# --- LEGACY LOGGER (for GUI/PID) ---
LOGGING = False
fieldnames = [
    "time",
    "motor_angle",
    "motor_target",
    "pendulum_angle",
    "pendulum_target",
    "rpm",
    "rpm_target",
    "voltage",
    "current",
]

directory = os.path.join(os.curdir, "Data")
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
filename = f"Data/log{files}.csv"
startTime = time.time()

def enableLogging():
    with open(filename, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    global LOGGING
    LOGGING = True

def save_data(data):
    if not LOGGING:
        return
    elapsedTime = round(time.time() - startTime, 3)
    with open(filename, "a", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        info = {
            "time": elapsedTime,
            "motor_angle": data[0],
            "motor_target": data[1],
            "pendulum_angle": data[2],
            "pendulum_target": data[3],
            "rpm": data[4],
            "rpm_target": data[5],
            "voltage": data[6],
            "current": data[7],
        }
        csv_writer.writerow(info)

# --- ASYNC LOGGER (for RL Deployment) ---
class AsyncLogger:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.queue = queue.Queue()
        self.filename = filename
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "dt", "theta", "alpha", "th_dot", "al_dot", "voltage", "mode"])
            while not (self.stop_event.is_set() and self.queue.empty()):
                try:
                    data = self.queue.get(timeout=0.1)
                    writer.writerow(data)
                except queue.Empty:
                    continue

    def log(self, data):
        self.queue.put(data)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
