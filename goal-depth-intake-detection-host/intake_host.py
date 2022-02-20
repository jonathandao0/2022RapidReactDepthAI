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
parser.add_argument('--demo', dest='demo', action="store_true", default=False, help='Enable Demo Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class IntakeHost:
    run_thread = None
    valid_labels = []
    null_labels = []
    target_lock = False
    target_id = None

    def __init__(self):
        log.debug("Connected Devices:")
        for device in dai.Device.getAllAvailableDevices():
            log.debug(f"{device.getMxId()} {device.state}")

        self.init_networktables()
        self.nt_controls = NetworkTables.getTable("Controls")
        self.nt_vision = NetworkTables.getTable("Vision")

        self.device_info = {
            'name': "OAK-1_Intake",
            'valid_ids': ["14442C10C14F47D700",
                          "14442C1011043ED700",
                          "184430105169091300"],
            'id': None,
            'fps_handler': FPSHandler(),
            'nt_tab': NetworkTables.getTable("OAK-1_Intake")
        }

        self.intake_pipeline, self.intake_labels = object_tracker.create_pipeline(MODEL_NAME)

        self.oak_1_stream = ImageZMQClient("camera 1", 5809, resolution=(480, 270))

    def parse_intake_frame(self, frame, bboxes, counters):
        # edgeFrame = cv2.threshold(edgeFrame, 60, 255, cv2.THRESH_TOZERO)[1]

        x_min_b = int((1920 / 4 - 1080 / 4) / 2.0)
        y_min_b = int((1080 / 4 - 1080 / 4) / 2.0)
        x_max_b = int((1920 / 4 - 1080 / 4) / 2.0 + (1080 / 4))
        y_max_b = int((1080 / 4 - 1080 / 4) / 2.0 + (1080 / 4))
        cv2.rectangle(frame, (x_min_b, y_min_b), (x_max_b, y_max_b), (0, 0, 0), 2)

        alliance_color = self.nt_controls.getString("alliance_string", "Invalid")
        tracking_type = self.nt_vision.getNumber("intake_tracking_type", 0)
        self.target_lock = self.nt_vision.getNumber("intake_target_lock", 0)

        if tracking_type == 1:
            if alliance_color.lower() == "red":
                self.valid_labels = ['red_launchpad']
            elif alliance_color.lower() == "blue":
                self.valid_labels = ['blue_launchpad']
            else:
                self.valid_labels = ['red_launchpad', 'blue_launchpad']
            self.null_labels = []
        else:
            if alliance_color.lower() == "red":
                self.valid_labels = ['red_cargo']
                self.null_labels = ['blue_cargo']
            elif alliance_color.lower() == "blue":
                self.valid_labels = ['blue_cargo']
                self.null_labels = ['red_cargo']
            else:
                self.valid_labels = ['red_cargo', 'blue_cargo']
                self.null_labels = []

        nt_tab = self.device_info['nt_tab']

        filtered_bboxes = []
        null_bboxes = []
        for bbox in bboxes:
            if self.intake_labels[bbox['label']] in self.valid_labels:
                filtered_bboxes.append(bbox)
            if self.intake_labels[bbox['label']] in self.null_labels:
                null_bboxes.append(bbox)

        filtered_bboxes.sort(key=operator.itemgetter('size'), reverse=True)

        if len(filtered_bboxes) == 0:
            nt_tab.putNumber("tv", 0)
            nt_tab.putNumberArray("ta", [0])
        else:
            nt_tab.putNumber("tv", 1)

        if self.target_lock and self.target_id is None:
            self.target_id = filtered_bboxes[0]['id']
        elif not self.target_lock:
            self.target_id = None

        if self.target_lock and self.target_id is not None:
            for i, bbox in filtered_bboxes:
                if bbox['id'] == self.target_id:
                    filtered_bboxes.insert(0, filtered_bboxes.pop(filtered_bboxes.index(i)))
                    break

        target_angles = []
        for bbox in filtered_bboxes:
            angle_offset = (bbox['x_mid'] - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920
            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (0, 255, 0), 2)

            if args.demo:
                cv2.putText(frame, "{}".format(self.intake_labels[bbox['label']]), (bbox['x_min'], bbox['y_min'] + 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "{}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            target_angles.append(angle_offset)
            bbox['angle_offset'] = angle_offset

        nt_tab.putNumberArray("ta", target_angles)

        for bbox in null_bboxes:
            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), (125, 125, 125), 2)

            if args.demo:
                cv2.putText(frame, "{}".format(self.intake_labels[bbox['label']]), (bbox['x_min'], bbox['y_min'] + 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, "{}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35),  (0, 0, 0), -1)
        if tracking_type == 1:
            if alliance_color.lower() == "red":
                color = (0, 0, 255)
            elif alliance_color.lower() == "blue":
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)

            cv2.putText(frame, "Tracking Launchpad", (60, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            red_offset = nt_tab.getNumber("red_counter_offset", 0)
            blue_offset = nt_tab.getNumber("blue_counter_offset", 0)
            red_count = counters['red_cargo']
            blue_count = counters['blue_cargo']

            cv2.putText(frame, "RED:{:.1s}".format(str(red_count - red_offset)), (80, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "BLUE:{:.1s}".format(str(blue_count - blue_offset)), (200, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

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
                    log.info("Intake Camera {} found".format(self.device_info['id']))
                    break

            if not found:
                log.error("No Intake Cameras found. Polling again in 5 seconds...")
                sleep(5)

        while True:
            if self.run_thread is None or not self.run_thread.is_alive():
                found, device = dai.Device.getDeviceByMxId(self.device_info['id'])
                self.device_info['nt_tab'].putBoolean("OAK-1_Intake Status", found)

                if found:
                    log.info("Intake Camera {} found. Starting processing thread...".format(self.device_info['id']))

                    self.run_thread = threading.Thread(target=self.run_intake_detection, args=(device,))
                    self.run_thread.daemon = True
                    self.run_thread.start()
                else:
                    log.error("Intake Camera {} not found. Attempting to restart thread...".format(self.device_info['id']))

            if self.run_thread is not None and not self.run_thread.is_alive():
                log.error("Intake thread died. Restarting thread...")

            sleep(1)

    def run_intake_detection(self, device):
        try:
            for frame, bboxes, counters in object_tracker.capture(device):
                self.parse_intake_frame(frame, bboxes, counters)
        except Exception as e:
            log.error("Exception {}".format(e))


class IntakeHostDebug(IntakeHost):

    def __init__(self):
        super().__init__()
        log.setLevel(level=logging.DEBUG)

    def parse_intake_frame(self, frame,  bboxes, counters):
        frame, bboxes, counters = super().parse_intake_frame(frame, bboxes, counters)

        for i, bbox in enumerate(bboxes):
            angle_offset = bbox['angle_offset'] if 'angle_offset' in bbox else 0

            frame_color = (0, 255, 0) if i == 0 else (0, 150, 150)

            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), frame_color, 2)
            cv2.putText(frame, "label: {}".format(self.intake_labels[bbox['label']]),
                        (bbox['x_min'], bbox['y_min'] + 30),
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
