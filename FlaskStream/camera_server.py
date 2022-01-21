from base_camera import BaseCamera


class CameraServer(BaseCamera):
    def __init__(self, feed_type, device, image_hub):
        super(CameraServer, self).__init__(feed_type, device, image_hub)

    @classmethod
    def server_frames(cls, image_hub):
        while True:

            cam_id, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')

            yield cam_id, frame
