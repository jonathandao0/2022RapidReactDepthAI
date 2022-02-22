#!/usr/bin/env python3

import argparse
import threading
import numpy as np
from time import sleep

import cv2
import depthai as dai

from PIL import Image
from FlaskStream.camera_client import ImageZMQClient
from common.config import NN_IMG_SIZE, MODEL_NAME

from pipelines import goal_depth_detection
import logging

from networktables.util import NetworkTables
from common.utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
parser.add_argument('--demo', dest='demo', action="store_true", default=False, help='Enable Demo Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


def label_frame(frame, bbox):
    cv2.putText(frame, "x: {}".format(round(bbox['x_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 30),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, "z: {}".format(round(bbox['depth_z'], 2)), (bbox['x_min'], bbox['y_min'] + 70),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, "h_angle: {}".format(round(bbox['h_angle'], 3)), (bbox['x_min'], bbox['y_min'] + 90),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, "v_angle: {}".format(round(bbox['v_angle'], 3)), (bbox['x_min'], bbox['y_min'] + 110),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    # cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 130),
    #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    # cv2.putText(frame, "label: {}".format(self.goal_labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 150),
    #             cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))


class GoalHost:
    run_thread = None

    def __init__(self):
        log.debug("Connected Devices:")
        for device in dai.Device.getAllAvailableDevices():
            log.debug(f"{device.getMxId()} {device.state}")

        self.init_networktables()
        self.nt_controls = NetworkTables.getTable("Controls")

        self.device_info = {
            'name': "OAK-D_Goal",
            'valid_ids': ["184430105169091300",
                          "18443010B1FA0C1300",
                          "18443010A1D0AA1200"],
            'id': None,
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-D_Goal")
        }

        self.goal_pipeline, self.goal_labels = goal_depth_detection.create_pipeline(MODEL_NAME)

        self.oak_d_stream = ImageZMQClient("camera 0", 5808, resolution=None)

    def parse_goal_frame(self, frame, bboxes):
        valid_labels = ['upper_hub']

        nt_tab = self.device_info['nt_tab']

        # edgeFrame = cv2.threshold(edgeFrame, 60, 255, cv2.THRESH_TOZERO)[1]
        # kernel = np.ones((3, 3), np.uint8)
        # edgeFrame = cv2.morphologyEx(edgeFrame, cv2.MORPH_CLOSE, kernel, iterations=1)

        if len(bboxes) == 0:
            nt_tab.putString("target_label", "None")
            nt_tab.putNumber("tv", 0)
            nt_tab.putNumber("tx", 0)
            nt_tab.putNumber("ty", 0)
            nt_tab.putNumber("tz", 0)
        else:
            for bbox in bboxes:
                target_label = self.goal_labels[bbox['label']]
                if target_label not in valid_labels:
                    continue

                # edgeFrame, target_x, target_y = target_finder.find_largest_contour(edgeFrame, bbox)

                # if target_x == -999 or target_y == -999:
                #     log.error("Error: Could not find target contour")
                #     continue
                target_x = bbox['x_mid']
                target_y = bbox['y_mid']

                horizontal_angle_offset = (target_x - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / NN_IMG_SIZE
                vertical_angle_offset = -(target_y - (NN_IMG_SIZE / 2.0)) * 38.6965126991271 / 234

                if abs(horizontal_angle_offset) > 30:
                    log.debug("Invalid angle offset. Setting it to 0")
                    nt_tab.putNumber("tv", 0)
                    horizontal_angle_offset = 0
                else:
                    log.debug("Found target '{}'\tX Angle Offset: {}".format(target_label, horizontal_angle_offset))
                    nt_tab.putNumber("tv", 1)

                if abs(horizontal_angle_offset) > 30 and abs(vertical_angle_offset) > 30:
                    log.debug("Target not valid for distance measurements")
                    nt_tab.putNumber("tg", 0)
                else:
                    nt_tab.putNumber("tg", 1)

                nt_tab.putString("target_label", target_label)
                nt_tab.putNumber("tx", horizontal_angle_offset)
                nt_tab.putNumber("ty", vertical_angle_offset)
                nt_tab.putNumber("tz", bbox['depth_z'])

                # cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                #               (255, 255, 255), 2)
                cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (255, 255, 255), 2)

                # cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128),
                #            thickness=-1)

                # bbox['target_x'] = target_x
                # bbox['target_y'] = target_y
                bbox['h_angle'] = horizontal_angle_offset
                bbox['v_angle'] = vertical_angle_offset

                if args.demo:
                    label_frame(frame, bbox)

        fps = self.device_info['fps_handler']
        fps.nextIter()

        # cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        if not args.demo:
            # gray_frame = Image.fromarray(edgeFrame, 'L')
            # gray_frame = cv2.cvtColor(edgeFrame, cv2.COLOR_GRAY2RGB)

            self.oak_d_stream.send_frame(frame)
        else:
            self.oak_d_stream.send_frame(frame)

        return frame, bboxes

    def init_networktables(self):
        NetworkTables.startClientTeam(4201)

        if not NetworkTables.isConnected():
            log.debug("Could not connect to team client. Trying other addresses...")
            NetworkTables.startClient([
                '10.42.1.2',
                '127.0.0.1',
                '10.0.0.2',
                '192.168.100.108'
            ])

        if NetworkTables.isConnected():
            log.debug("NT Connected to {}".format(NetworkTables.getRemoteAddress()))
            return True
        else:
            log.error("Could not connect to NetworkTables. Restarting server...")
            return False

    def run(self):
        log.debug("Setup complete, parsing frames...")

        found = False
        while self.device_info['id'] is None:
            for device_id in self.device_info['valid_ids']:
                found, device = dai.Device.getDeviceByMxId(device_id)

                if found:
                    self.device_info['id'] = device_id
                    log.info("Goal Camera {} found".format(self.device_info['id']))
                    break

            if not found:
                log.error("No Goal Cameras found. Polling again...")
                sleep(1)

        while True:
            if self.run_thread is None or not self.run_thread.is_alive():
                found, device = dai.Device.getDeviceByMxId(self.device_info['id'])
                self.device_info['nt_tab'].putBoolean("OAK-D_Goal Status", found)

                if found:
                    log.info("Goal Camera {} found. Starting processing thread...".format(self.device_info['id']))

                    self.run_thread = threading.Thread(target=self.run_goal_detection, args=(device,))
                    self.run_thread.daemon = True
                    self.run_thread.start()
                else:
                    log.error("Goal Camera {} not found. Attempting to restart thread...".format(self.device_info['id']))

            if self.run_thread is not None and not self.run_thread.is_alive():
                log.error("Goal thread died. Restarting thread...")

            sleep(1)

    def run_goal_detection(self, device):
        try:
            for frame, bboxes in goal_depth_detection.capture(device):
                self.parse_goal_frame(frame, bboxes)
        except Exception as e:
            log.error("Exception {}".format(e))


class GoalHostDebug(GoalHost):

    def __init__(self):
        super().__init__()
        log.setLevel(logging.DEBUG)

    def parse_goal_frame(self, frame, bboxes):
        frame, bboxes = super().parse_goal_frame(frame, bboxes)
        valid_labels = ['upper_hub']

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            # label_frame(edgeFrame, bbox)

            if not args.demo:
                label_frame(frame, bbox)

        # cv2.imshow("OAK-D Goal Edge", edgeFrame)
        cv2.imshow("OAK-D Goal ", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    log.info("Starting goal-depth-detection-host")
    if args.debug:
        GoalHostDebug().run()
    else:
        GoalHost().run()
