import logging

import numpy as np

from common.config import NN_IMG_SIZE

log = logging.getLogger(__name__)

class VideoFrameHandler:
    empty_frame = np.zeros((NN_IMG_SIZE, NN_IMG_SIZE))
    left_frame = empty_frame
    right_frame = empty_frame
    frame_to_send = np.concatenate((empty_frame, empty_frame), axis=1)

    def __init__(self):
        # self.run()
        pass

    def run(self):
        while True:
            self.process_frames()

    def send_left_frame(self, frame):
        self.left_frame = frame

    def send_right_frame(self, frame):
        self.right_frame = frame

    def process_frames(self):
        try:
            # self.frame_to_send = np.hstack((self.left_frame, self.right_frame))
            self.frame_to_send = self.right_frame
        except Exception as e:
            log.error("Error Processing frame {}".format(e))

        return self.frame_to_send
