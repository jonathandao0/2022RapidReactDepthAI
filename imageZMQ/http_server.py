import argparse
import logging

import cv2
import sys
import imagezmq
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='hostname', default='localhost', help='Set hostname (default: localhost)')
parser.add_argument('-p', dest='port', default=5802, type=int, help='Set port (default 5802)')
args = parser.parse_args()


def sendImagesToWeb():
    receiver = imagezmq.ImageHub(open_port='tcp://127.0.0.1:5566', REQ_REP=False)
    while True:
        camName, frame = receiver.recv_image()
        jpg = cv2.imencode('.jpg', frame)[1]
        yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n' + jpg.tostring() + b'\r\n'


@Request.application
def application(request):
    return Response(sendImagesToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    hostname = args.hostname
    port = args.port
    log.info("Starting ImageZMQ HTTP Server at {}:{}".format(hostname, port))

    run_simple(hostname, port, application)
