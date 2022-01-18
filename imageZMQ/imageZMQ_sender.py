import sys
import threading

import numpy as np
import time
import imagezmq


class ImageZMQSender:
    image_to_send = None

    def run(self):
        while True:  # press Ctrl-C to stop image sending program
            # Send an image to the queue
            if self.image_to_send is not None:
                self.sender.send_image(self.feedName, self.image_to_send)
                time.sleep(1.0/self.FPS)

    def send_image(self, image):
        self.image_to_send = image

    def __init__(self, FEED_NAME='Raspberry Pi 4', HOSTNAME='localhost', HTTP_PORT=5555, FPS=30):
        self.feedName = FEED_NAME
        self.FPS = FPS
        connection_address = 'tcp://{}:{}'.format(HOSTNAME, HTTP_PORT)

        # Create an image sender in PUB/SUB (non-blocking) mode
        self.sender = imagezmq.ImageSender(connect_to=connection_address)

        th = threading.Thread(target=self.run)
        th.daemon = True
        th.start()
