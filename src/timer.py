import time

class Timer:
    def __init__(self):
        self.start_time = 0
        self.time = 0

    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.time = time.time() - self.start_time