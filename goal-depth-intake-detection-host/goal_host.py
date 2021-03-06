#!/usr/bin/env python3

import argparse
import math
import threading
import platform
from operator import itemgetter

import numpy as np
from time import sleep

import cv2
import depthai as dai

from PIL import Image
from FlaskStream.camera_client import ImageZMQClient
from common.config import NN_IMG_SIZE, MODEL_NAME

from pipelines import goal_depth_detection, goal_depth_detection_recording
import logging

from networktables.util import NetworkTables
from common.utils import FPSHandler

if platform.system() != 'Windows':
    from CSCoreStream.cscore_client import CsCoreClient

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
parser.add_argument('--demo', dest='demo', action="store_true", default=False, help='Enable Demo Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)

ENABLE_RECORDING = False


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
            'valid_ids': [#"184430105169091300",
                          #"18443010B1FA0C1300",
                          "18443010A1D0AA1200",
                          "14442C1091398FD000",
                          ],
            'id': None,
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-D_Goal")
        }

        if ENABLE_RECORDING:
            self.goal_pipeline, self.goal_labels = goal_depth_detection_recording.create_pipeline(MODEL_NAME)
        else:
            self.goal_pipeline, self.goal_labels = goal_depth_detection.create_pipeline(MODEL_NAME)

        if platform.system() == 'Windows':
            self.oak_d_stream = ImageZMQClient("camera 0", 5808, resolution=None)
        else:
            self.oak_d_stream = CsCoreClient("camera 0", 5808, resolution=(200, 200))

    def parse_goal_frame(self, frame, bboxes, metadata):
        valid_labels = ['upper_hub', 'lower_hub']

        nt_tab = self.device_info['nt_tab']

        # edgeFrame = cv2.threshold(edgeFrame, 60, 255, cv2.THRESH_TOZERO)[1]
        # kernel = np.ones((3, 3), np.uint8)
        # edgeFrame = cv2.morphologyEx(edgeFrame, cv2.MORPH_CLOSE, kernel, iterations=1)

        filtered_bboxes = []
        if len(bboxes) == 0:
            nt_tab.putString("target_label", "None")
            nt_tab.putNumber("tv", 0)
            nt_tab.putNumber("tx", 0)
            nt_tab.putNumber("ty", 0)
            nt_tab.putNumber("tz", 0)
        else:
            if len(bboxes) > 0:
                for bbox in bboxes:
                    if self.goal_labels[bbox['label']] in valid_labels:
                        filtered_bboxes.append(bbox)

                if len(filtered_bboxes) > 1:
                    filtered_bboxes.sort(key=lambda x: x['confidence'], reverse=True)
                    filtered_bboxes.sort(key=lambda x: x['label'])

                for i, bbox in enumerate(filtered_bboxes):
                    # edgeFrame, target_x, target_y = target_finder.find_largest_contour(edgeFrame, bbox)

                    # if target_x == -999 or target_y == -999:
                    #     log.error("Error: Could not find target contour")
                    #     continue
                    target_x = bbox['x_mid']
                    target_y = bbox['y_mid']

                    # Pinhole camera model. See 254's 2016 vision talk
                    horizontal_angle_radians = math.atan((target_x - (NN_IMG_SIZE / 2.0)) / (NN_IMG_SIZE / (2 * math.tan(math.radians(69.0) / 2))))
                    horizontal_angle_offset = math.degrees(horizontal_angle_radians)
                    vertical_angle_radians = -math.atan((target_y - (NN_IMG_SIZE / 2.0)) / (NN_IMG_SIZE / (2 * math.tan(math.radians(54.0) / 2))))
                    vertical_angle_offset = math.degrees(vertical_angle_radians)

                    if abs(horizontal_angle_offset) > 40:
                        log.debug("Invalid angle offset. Setting it to 0")
                        nt_tab.putNumber("tv", 0)
                        horizontal_angle_offset = 0
                    else:
                        log.debug("Found target '{}'\tX Angle Offset: {}".format(self.goal_labels[bbox['label']], horizontal_angle_offset))
                        nt_tab.putNumber("tv", 1 if self.goal_labels[bbox['label']] == 'upper_hub' else 2)

                    if abs(horizontal_angle_offset) > 40 and abs(vertical_angle_offset) > 30:
                        log.debug("Target not valid for distance measurements")
                        nt_tab.putNumber("tg", 0)
                    else:
                        nt_tab.putNumber("tg", 1)

                    color = (128, 128, 128)
                    if i == 0:
                        nt_tab.putString("target_label", self.goal_labels[bbox['label']])
                        nt_tab.putNumber("tx", horizontal_angle_offset)
                        nt_tab.putNumber("ty", vertical_angle_offset)
                        nt_tab.putNumber("tz", bbox['depth_z'])
                        nt_tab.putNumber("timestamp", metadata['timestamp'].total_seconds())
                        NetworkTables.flush()
                        color = (0, 255, 0)

                    # cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                    #               (255, 255, 255), 2)

                    cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), color, 2)

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
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

        # if not args.demo:
        #     # gray_frame = Image.fromarray(edgeFrame, 'L')
        #     # gray_frame = cv2.cvtColor(edgeFrame, cv2.COLOR_GRAY2RGB)
        #     output_frame = frame[54:324, 0:NN_IMG_SIZE]
        #     self.oak_d_stream.send_frame(output_frame)
        # else:
        # output_frame = frame[91:324, 0:NN_IMG_SIZE]
        self.oak_d_stream.send_frame(frame)

        return frame, filtered_bboxes, metadata

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
            for frame, bboxes, metadata in goal_depth_detection.capture(device):
                self.parse_goal_frame(frame, bboxes, metadata)
        except Exception as e:
            log.error("Exception {}".format(e))


class GoalHostDebug(GoalHost):

    def __init__(self):
        super().__init__()
        log.setLevel(logging.DEBUG)

    def parse_goal_frame(self, frame, bboxes, metadata):
        frame, bboxes, metadata = super().parse_goal_frame(frame, bboxes, metadata)
        valid_labels = ['upper_hub', 'lower_hub']

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
