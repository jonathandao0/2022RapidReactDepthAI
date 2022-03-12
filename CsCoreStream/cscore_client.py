import cscore
import logging
import threading

import cv2
import imutils

log = logging.getLogger(__name__)


class CsCoreClient:
    frame_to_send = None

    def __init__(self, camera_id, port, resolution=(320, 320), fps_limit=30):
        self.camera_id = camera_id
        self.port = port
        self.resolution = resolution
        self.fps_limit = fps_limit

        self.camera = cscore.CvSource(camera_id, cscore.VideoMode.PixelFormat.kMJPEG, resolution[0], resolution[1], fps_limit)

        th = threading.Thread(target=self.run)
        th.daemon = True
        th.start()

        self.mjpegServer = cscore.MjpegServer("httpserver", self.port)
        self.mjpegServer.setSource(self.camera)

    def send_frame(self, frame):
        self.frame_to_send = frame

    def run(self):
        while True:
            if self.frame_to_send is not None:

                if self.resolution is not None:
                    resized_frame = imutils.resize(self.frame_to_send, self.resolution[0], self.resolution[1], inter=cv2.INTER_LINEAR)
                else:
                    resized_frame = self.frame_to_send

                self.camera.putFrame(resized_frame)
