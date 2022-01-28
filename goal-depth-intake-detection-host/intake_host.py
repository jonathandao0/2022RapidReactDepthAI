#!/usr/bin/env python3

import argparse
import operator
import threading
import numpy as np
from time import sleep

import cv2
import depthai as dai

from FlaskStream.camera_client import ImageZMQClient
from common.config import NN_IMG_SIZE, MODEL_NAME

from pipelines import object_tracker
import logging

from networktables.util import NetworkTables
from common.utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class IntakeHost:
    run_thread = None

    def __init__(self):
        log.info("Connected Devices:")
        for device in dai.Device.getAllAvailableDevices():
            log.info(f"{device.getMxId()} {device.state}")

        self.init_networktables()
        self.nt_controls = NetworkTables.getTable("Controls")

        self.device_info = {
            'name': "OAK-1_Intake",
            'id': "14442C10C14F47D700",
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-1_Intake")
        }

        self.intake_pipeline, self.intake_labels = object_tracker.create_pipeline(MODEL_NAME)

        self.oak_1_stream = ImageZMQClient("camera 1", 5809)

    def parse_intake_frame(self, frame, bboxes, counters):
        # edgeFrame = cv2.threshold(edgeFrame, 60, 255, cv2.THRESH_TOZERO)[1]

        alliance_color = self.nt_controls.getString("Alliance", "Invalid")

        if alliance_color == "Red":
            valid_labels = ['red_cargo']
            null_labels = ['blue_cargo']
        elif alliance_color == "Blue":
            valid_labels = ['blue_cargo']
            null_labels = ['red_cargo']
        else:
            valid_labels = ['red_cargo', 'blue_cargo']
            null_labels = []

        nt_tab = self.device_info['nt_tab']

        filtered_bboxes = []
        null_bboxes = []
        for bbox in bboxes:
            if self.intake_labels[bbox['label']] in valid_labels:
                filtered_bboxes.append(bbox)
            if self.intake_labels[bbox['label']] in null_labels:
                null_bboxes.append(bbox)

        filtered_bboxes.sort(key=operator.itemgetter('size'), reverse=True)

        if len(filtered_bboxes) == 0:
            nt_tab.putNumber("tv", 0)
            nt_tab.putNumberArray("ta", [0])
        else:
            nt_tab.putNumber("tv", 1)

            target_angles = []
            for bbox in filtered_bboxes:
                angle_offset = (bbox['x_mid'] - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920
                cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (0, 255, 0), 2)

                target_angles.append(angle_offset)
                bbox['angle_offset'] = angle_offset

            nt_tab.putNumberArray("ta", target_angles)

            for bbox in null_bboxes:
                cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (255, 0, 0), 2)

        cv2.rectangle(frame, (0, 0), (NN_IMG_SIZE, 35),  (0, 0, 0), -1)

        red_offset = nt_tab.getNumber("red_counter_offset", 0)
        blue_offset = nt_tab.getNumber("blue_counter_offset", 0)
        red_count = counters['red_cargo']
        blue_count = counters['blue_cargo']

        cv2.putText(frame, "RED:{:.1s}".format(str(red_count - red_offset)), (80, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "BLUE:{:.1s}".format(str(blue_count - blue_offset)), (200, 28), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        nt_tab.putNumber("red_count", red_count)
        nt_tab.putNumber("blue_count", blue_count)

        fps = self.device_info['fps_handler']
        fps.nextIter()
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        self.oak_1_stream.send_frame(frame)

        return frame, filtered_bboxes, counters

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
                    self.device_info['nt_tab'].putBoolean("OAK-1_Intake Status", found)

                    if found:
                        self.run_thread = threading.Thread(target=self.run_intake_detection, args=(device_id,))
                        self.run_thread.daemon = True
                        self.run_thread.start()

                sleep(1)
            except Exception as e:
                log.error("Exception {}".format(e))

    def run_intake_detection(self, device_id):
        for frame, bboxes, counters in object_tracker.capture(device_id):
            self.parse_intake_frame(frame, bboxes, counters)


class IntakeHostDebug(IntakeHost):

    def __init__(self):
        super().__init__()

    def parse_intake_frame(self, frame,  bboxes, counters):
        frame, bboxes, counters = super().parse_intake_frame(frame, bboxes, counters)

        for i, bbox in enumerate(bboxes):
            angle_offset = bbox['angle_offset'] if 'angle_offset' in bbox else 0

            frame_color = (0, 255, 0) if i == 0 else (0, 150, 150)

            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), frame_color, 2)
            cv2.putText(frame, "label: {}".format(self.intake_labels[bbox['label']]), (bbox['x_min'], bbox['y_min'] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "x: {}".format(round(bbox['x_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "angle: {}".format(round(angle_offset, 3)), (bbox['x_min'], bbox['y_min'] + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "size: {}".format(round(bbox['size'], 3)), (bbox['x_min'], bbox['y_min'] + 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 130),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        # cv2.imshow("OAK-1 Intake Edge", edgeFrame)
        cv2.imshow("OAK-1 Intake", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    log.info("Starting IntakeHost")
    if args.debug:
        IntakeHostDebug().run()
    else:
        IntakeHost().run()
