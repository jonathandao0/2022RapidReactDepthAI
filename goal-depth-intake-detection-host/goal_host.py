#!/usr/bin/env python3

import argparse
import threading
import numpy as np
from time import sleep

import cv2
import depthai as dai

from FlaskStream.camera_client import ImageZMQClient
from common.config import NN_IMG_SIZE, MODEL_NAME

from pipelines import goal_edge_depth_detection, object_tracker
import logging
from common import target_finder

from networktables.util import NetworkTables
from common.utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class GoalHost:
    run_thread = None

    def __init__(self):
        log.info("Connected Devices:")
        for device in dai.Device.getAllAvailableDevices():
            log.info(f"{device.getMxId()} {device.state}")

        self.init_networktables()
        self.nt_controls = NetworkTables.getTable("Controls")

        self.device_info = {
            'name': "OAK-D_Goal",
            'id': "184430105169091300",
            # 'id': "14442C1091398FD000",
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-D_Goal")
        }

        self.goal_pipeline, self.goal_labels = goal_edge_depth_detection.create_pipeline(MODEL_NAME)

        self.oak_d_stream = ImageZMQClient("camera 1", 5808)

    def parse_goal_frame(self, frame, edgeFrame, bboxes):
        valid_labels = ['upper_hub']

        nt_tab = self.device_info['nt_tab']

        if len(bboxes) == 0:
            nt_tab.putString("target_label", "None")
            nt_tab.putNumber("tv", 0)
        else:
            for bbox in bboxes:
                target_label = self.goal_labels[bbox['label']]
                if target_label not in valid_labels:
                    continue

                edgeFrame, target_x, target_y = target_finder.find_largest_contour(edgeFrame, bbox)

                if target_x == -999 or target_y == -999:
                    log.error("Error: Could not find target contour")
                    continue

                horizontal_angle_offset = (target_x - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920
                vertical_angle_offset = (target_y - (NN_IMG_SIZE / 2.0)) * 38.6965126991271 / 1080

                if abs(horizontal_angle_offset) > 30:
                    log.info("Invalid angle offset. Setting it to 0")
                    nt_tab.putNumber("tv", 0)
                    horizontal_angle_offset = 0
                else:
                    log.info("Found target '{}'\tX Angle Offset: {}".format(target_label, horizontal_angle_offset))
                    nt_tab.putNumber("tv", 1)

                if abs(horizontal_angle_offset) > 30 and abs(vertical_angle_offset) > 30:
                    log.info("Target not valid for distance measurements")
                    nt_tab.putNumber("tg", 0)
                else:
                    nt_tab.putNumber("tg", 1)

                nt_tab.putString("target_label", target_label)
                nt_tab.putNumber("tx", horizontal_angle_offset)
                nt_tab.putNumber("ty", vertical_angle_offset)
                nt_tab.putNumber("tz", bbox['depth_z'])

                cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (255, 255, 255), 2)

                # cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128),
                #            thickness=-1)

                bbox['target_x'] = target_x
                bbox['target_y'] = target_y
                bbox['horizontal_angle_offset'] = horizontal_angle_offset

        fps = self.device_info['fps_handler']
        fps.next_iter()
        cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        bgr_frame = cv2.cvtColor(edgeFrame, cv2.COLOR_GRAY2RGB)
        self.oak_d_stream.send_frame(bgr_frame)

        return frame, edgeFrame, bboxes

    def init_networktables(self):
        NetworkTables.startClientTeam(4201)

        if not NetworkTables.isConnected():
            log.info("Could not connect to team client. Trying other addresses...")
            NetworkTables.startClient([
                '10.42.1.2',
                '127.0.0.1',
                '10.0.0.2',
                '192.168.100.108'
            ])

        if NetworkTables.isConnected():
            log.info("NT Connected to {}".format(NetworkTables.getRemoteAddress()))
            return True
        else:
            log.error("Could not connect to NetworkTables. Restarting server...")
            return False

    def run(self):
        log.info("Setup complete, parsing frames...")

        while True:
            try:
                if self.run_thread is None or not self.run_thread.is_alive():
                    found, device_id = dai.Device.getDeviceByMxId(self.device_info['id'])
                    self.device_info['nt_tab'].putBoolean("OAK-D_Goal Status", found)

                    if found:
                        self.run_thread = threading.Thread(target=self.run_goal_detection, args=(device_id,))
                        self.run_thread.daemon = True
                        self.run_thread.start()

                sleep(1)
            except Exception as e:
                log.error("Exception {}".format(e))

    def run_goal_detection(self, device_id):
        for frame, edgeFrame, bboxes in goal_edge_depth_detection.capture(device_id):
            self.parse_goal_frame(frame, edgeFrame, bboxes)


class GoalHostDebug(GoalHost):

    def __init__(self):
        super().__init__()

    def parse_goal_frame(self, frame, edgeFrame, bboxes):
        frame, edgeFrame, bboxes = super().parse_goal_frame(frame, edgeFrame, bboxes)
        valid_labels = ['upper_hub']

        for bbox in bboxes:
            target_label = self.goal_labels[bbox['label']]

            if target_label not in valid_labels:
                continue

            target_x = bbox['target_x'] if 'target_x' in bbox else 0
            angle_offset = bbox['angle_offset'] if 'angle_offset' in bbox else 0

            cv2.putText(edgeFrame, "x: {}".format(round(target_x, 2)), (bbox['x_min'], bbox['y_min'] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "z: {}".format(round(bbox['depth_z'], 2)), (bbox['x_min'], bbox['y_min'] + 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "angle: {}".format(round(angle_offset, 3)), (bbox['x_min'], bbox['y_min'] + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(edgeFrame, "label: {}".format(self.goal_labels[bbox['label']], 1), (bbox['x_min'], bbox['y_min'] + 130),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        cv2.imshow("OAK-D Goal Edge", edgeFrame)
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
