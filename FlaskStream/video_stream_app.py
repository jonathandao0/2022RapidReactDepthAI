import argparse
import logging
from importlib import import_module
from flask import Flask, render_template, Response
import cv2
import simplejpeg


app = Flask(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='hostname', default='localhost', help='Set hostname (default: localhost)')
parser.add_argument('-p', dest='port', default=5802, type=int, help='Set port (default 5802)')
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera_stream, feed_type, device):
    """Video streaming generator function."""
    unique_name = (feed_type, device)

    while True:
        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        frame = simplejpeg.encode_jpeg(frame, quality=100, colorspace='BGR', fastdct=True)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<feed_type>/<device>')
def video_feed(feed_type, device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    port_list = (5808, 5809)
    camera_stream = import_module('camera_server').CameraServer
    return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    hostname = args.hostname
    port = args.port
    debug = args.debug

    app.run(host=hostname, port=port, debug=debug, threaded=True)
