import argparse
import logging
from importlib import import_module
import socket

from flask import Flask, render_template, Response

import simplejpeg

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))

    ip_address = s.getsockname()[0]
except:
    ip_address = '10.42.1.100'

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='hostname', default=ip_address, help='Set hostname (default: localhost)')
parser.add_argument('-p', dest='port', default=5802, type=int, help='Set port (default: 5802)')
parser.add_argument('-q', dest='quality', type=int, default=100, help='Set jpeg quality (default: 100). Lower this to reduce bandwidth cost')
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera_stream, feed_type, device):
    global quality
    """Video streaming generator function."""
    unique_name = (feed_type, device)

    while True:
        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        frame = simplejpeg.encode_jpeg(frame, quality=quality, colorspace='BGR', colorsubsampling='420', fastdct=True)
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
    global quality
    hostname = args.hostname
    port = args.port
    debug = args.debug
    quality = args.quality

    log.info("Starting Flask app at {}:{}".format(hostname, port))

    app.run(host=hostname, port=port, debug=debug, threaded=True)
