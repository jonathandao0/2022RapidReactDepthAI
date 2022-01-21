import imagezmq


class ImageZMQClient:

    def __init__(self, camera_id, port):
        self.camera_id = camera_id
        self.sender = imagezmq.ImageSender(connect_to='tcp://localhost:{}'.format(port))

    def send_frame(self, frame):
        self.sender.send_image(self.camera_id, frame)
