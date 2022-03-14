import logging
import time
import threading

import cv2
import imagezmq
import imutils
import zmq

log = logging.getLogger(__name__)


class ImageZMQClient:
    frame_to_send = None

    def __init__(self, camera_id, port, resolution=(320, 320), fps_limit=30):
        self.camera_id = camera_id
        self.port = port
        self.sender = self.init_sender(self.port)
        self.resolution = resolution
        self.fps_limit = fps_limit

        th = threading.Thread(target=self.run)
        th.daemon = True
        th.start()

    def init_sender(self, port):
        sender = imagezmq.ImageSender(connect_to='tcp://localhost:{}'.format(port))
        sender.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
        # NOTE: because of the way PyZMQ and imageZMQ are implemented, the
        #       timeout values specified must be integer constants, not variables.
        #       The timeout value is in milliseconds, e.g., 2000 = 2 seconds.
        sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 20)  # set a receive timeout
        sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 20)  # set a send timeout

        return sender

    def send_frame(self, frame):
        self.frame_to_send = frame

        # Non-threaded code
        # if self.resolution is not None:
        #     frame = imutils.resize(frame, self.resolution[0], self.resolution[1], inter=cv2.INTER_LINEAR)
        # try:
        #     reply = self.sender.send_image(self.camera_id, frame)
        # except (zmq.ZMQError, zmq.ContextTerminated, zmq.Again):
        #     self.sender.close()
        #     log.debug('Restarting ImageSender.')
        #     self.sender = self.init_sender(self.port)

    def run(self):
        while True:
            if self.frame_to_send is not None:
                start_time = time.time()

                if self.resolution is not None:
                    resized_frame = imutils.resize(self.frame_to_send, self.resolution[0], self.resolution[1], inter=cv2.INTER_LINEAR)
                else:
                    resized_frame = self.frame_to_send

                try:
                    reply = self.sender.send_image(self.camera_id, resized_frame)
                except (zmq.ZMQError, zmq.ContextTerminated, zmq.Again):
                    self.sender.close()
                    log.debug('Restarting ImageSender.')
                    self.sender = self.init_sender(self.port)

                end_time = time.time()

                time.sleep(max(1./self.fps_limit - (end_time-start_time), 0))
            else:
                time.sleep(1)
