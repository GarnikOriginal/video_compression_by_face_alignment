import time


class FPSCounter:
    def __init__(self, frames_cup):
        self.start = None
        self.end = None
        self.fps = 0
        self.frames = 0
        self.frames_cup = frames_cup

    def run(self):
        self.start = time.time()

    def step(self):
        self.frames += 1
        self.end = time.time()
        self.fps = self.frames / (self.end - self.start)
        if self.frames == self.frames_cup:
            self.__reset_params__()

    def get_fps(self):
        return self.fps

    def __reset_params__(self):
        self.start = time.time()
        self.frames = 0
