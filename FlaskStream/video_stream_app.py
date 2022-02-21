import argparse
import logging
from importlib import import_module
import socket
from time import sleep

from flask import Flask, render_template, Response

import simplejpeg


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='hostname', default=None, help='Set hostname (default: localhost)')
parser.add_argument('-p', dest='port', default=5802, type=int, help='Set port (default: 5802)')
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
parser.add_argument('--demo', dest='demo', action="store_true", default=False, help='Enable Demo Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)

app = Flask(__name__)

if args.demo:
    CAMERA_SETTINGS = {
        'camera_0': {
            'colorspace': 'BGR',
            'quality': 80
        },
        'camera_1': {
            'colorspace': 'BGR',
            'quality': 80
        },
    }
else:
    CAMERA_SETTINGS = {
        'camera_0': {
            'colorspace': 'BGR',
            'quality': 30
        },
        'camera_1': {
            'colorspace': 'BGR',
            'quality': 30
        },
    }


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera_stream, feed_type, device):
    global CAMERA_SETTINGS
    """Video streaming generator function."""
    unique_name = (feed_type, device)
    name = '{}_{}'.format(feed_type, device)
    quality = CAMERA_SETTINGS[name]['quality']
    colorspace = CAMERA_SETTINGS[name]['colorspace']

    while True:
        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        frame = simplejpeg.encode_jpeg(frame, quality=quality, colorspace=colorspace, colorsubsampling='420', fastdct=True)
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
    while hostname is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))

            hostname = s.getsockname()[0]
        except:
            pass
        sleep(1)

    port = args.port
    debug = args.debug

    log.info("Starting Flask app at {}:{}".format(hostname, port))

    app.run(host=hostname, port=port, debug=debug, threaded=True)
