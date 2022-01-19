#!/usr/bin/env python3

import argparse
import operator
import threading
import numpy as np
from time import sleep

import cv2
import depthai as dai
import socket

from common.config import NN_IMG_SIZE, MODEL_NAME

from imageZMQ.imageZMQ_sender import ImageZMQSender
from imageZMQ.video_frame_handler import VideoFrameHandler
from pipelines import goal_edge_depth_detection, object_tracker
import logging
from common import target_finder

from networktables.util import NetworkTables
from common.utils import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
args = parser.parse_args()

log = logging.getLogger(__name__)


class Main:
    threadDict = {
        "OAK-D_Goal": None,
        "OAK-1_Intake": None,
        "VideoFrame": None
    }

    def __init__(self):
        log.info("Connected Devices:")
        for device in dai.Device.getAllAvailableDevices():
            log.info(f"{device.getMxId()} {device.state}")

        self.init_networktables()
        self.nt_controls = NetworkTables.getTable("Controls")

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))

            ip_address = s.getsockname()[0]
        except:
            ip_address = 'localhost'

        port1 = 5801
        port2 = 5802

        self.device_list = {"OAK-D_Goal": {
            'name': "OAK-D_Goal",
            'id': "184430105169091300",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port1),
            'nt_tab': NetworkTables.getTable("OAK-D_Goal")
        }, "OAK-1_Intake": {
            'name': "OAK-1_Intake",
            # 'id': "14442C1011043ED700",
            'id': "14442C1091398FD000",
            'fps_handler': FPSHandler(),
            'stream_address': "{}:{}".format(ip_address, port2),
            'nt_tab': NetworkTables.getTable("OAK-1_Intake")
        }}

        self.goal_pipeline, self.goal_labels = goal_edge_depth_detection.create_pipeline(MODEL_NAME)
        self.intake_pipeline, self.intake_labels = object_tracker.create_pipeline(MODEL_NAME)

        # self.oak_d_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port1, colorspace='BW', QUALITY=10)
        # self.oak_1_stream = MjpegStream(IP_ADDRESS=ip_address, HTTP_PORT=port2, colorspace='BW', QUALITY=10)
        # self.oak_1_stream = ImageZMQSender()
        # self.oak_d_stream = CsCoreStream(IP_ADDRESS=ip_address, HTTP_PORT=port1, colorspace='BW', QUALITY=10)
        # self.oak_1_stream = CsCoreStream(IP_ADDRESS=ip_address, HTTP_PORT=port2, colorspace='BW', QUALITY=10)

    def parse_goal_frame(self, frame, edgeFrame, bboxes):
        kernel = np.ones((3, 3), np.uint8)
        edgeFrame = cv2.morphologyEx(edgeFrame, cv2.MORPH_CLOSE, kernel, iterations=1)

        # edgeFrame = cv2.threshold(edgeFrame, 20, 255, cv2.THRESH_TOZERO)[1]

        valid_labels = ['upper_hub']

        nt_tab = self.device_list['OAK-D_Goal']['nt_tab']

        if len(bboxes) == 0:
            nt_tab.putString("target_label", "None")
            nt_tab.putNumber("tv", 0)
        else:
            for bbox in bboxes:
                target_label = self.goal_labels[bbox['label']]
                if target_label not in valid_labels:
                    continue

                edgeFrame, target_x, target_y = target_finder.find_largest_hexagon_contour(edgeFrame, bbox)

                if target_x == -999 or target_y == -999:
                    log.error("Error: Could not find target contour")
                    continue

                horizontal_angle_offset = (target_x - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920
                vertical_angle_offset = (target_y - (NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1080

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

                cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128),
                           thickness=-1)

                bbox['target_x'] = target_x
                bbox['target_y'] = target_y
                bbox['horizontal_angle_offset'] = horizontal_angle_offset

        fps = self.device_list['OAK-D_Goal']['fps_handler']
        fps.next_iter()
        cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        # self.video_frame_handler.send_left_frame(edgeFrame)

        return frame, edgeFrame, bboxes

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

        nt_tab = self.device_list['OAK-1_Intake']['nt_tab']

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

        red_offset = nt_tab.getNumber("red_counter_offset", 0)
        blue_offset = nt_tab.getNumber("blue_counter_offset", 0)
        red_count = counters['red_cargo']
        blue_count = counters['blue_cargo']

        cv2.putText(frame, "RED:{:.1s}".format(str(red_count - red_offset)), (80, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
        cv2.putText(frame, "BLUE:{:.1s}".format(str(blue_count - blue_offset)), (200, 28), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))

        nt_tab.putNumber("red_count", red_count)
        nt_tab.putNumber("blue_count", blue_count)

        fps = self.device_list['OAK-1_Intake']['fps_handler']
        fps.next_iter()
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (0, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        # self.video_frame_handler.send_right_frame(frame)

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
                # if self.threadDict['OAK-D_Goal'] is None or not self.threadDict['OAK-D_Goal'].is_alive():
                #     found, device_info = dai.Device.getDeviceByMxId(self.device_list['OAK-D_Goal']['id'])
                #     self.device_list['OAK-D_Goal']['nt_tab'].putBoolean("OAK-D_Goal Status", found)
                #
                #     if found:
                #         th = threading.Thread(target=self.run_goal_detection, args=(device_info,))
                #         th.start()
                #         self.threadDict['OAK-D_Goal'] = th

                if self.threadDict['OAK-1_Intake'] is None or not self.threadDict['OAK-1_Intake'].is_alive():
                    found, device_info = dai.Device.getDeviceByMxId(self.device_list['OAK-1_Intake']['id'])
                    self.device_list['OAK-1_Intake']['nt_tab'].putBoolean("OAK-1_Intake Status", found)

                    if found:
                        th = threading.Thread(target=self.run_intake_detection, args=(device_info,))
                        th.start()
                        self.threadDict['OAK-1_Intake'] = th

                # if self.threadDict['VideoFrame'] is None or not self.threadDict['VideoFrame'].is_alive():
                #     th = threading.Thread(target=self.video_frame_handler.run)
                #     th.start()
                #     self.threadDict['VideoFrame'] = th

                sleep(1)
            except Exception as e:
                log.error("Exception {}".format(e))

    def run_goal_detection(self, device_info):
        self.device_list['OAK-D_Goal']['nt_tab'].putString("OAK-D_Goal Stream", self.device_list['OAK-D_Goal']['stream_address'])
        for frame, edgeFrame, bboxes in goal_edge_depth_detection.capture(device_info):
            self.parse_goal_frame(frame, edgeFrame, bboxes)

    def run_intake_detection(self, device_info):
        self.device_list['OAK-1_Intake']['nt_tab'].putString("OAK-1 Stream", self.device_list['OAK-1_Intake']['stream_address'])
        for frame, bboxes, counters in object_tracker.capture(device_info):
            self.parse_intake_frame(frame, bboxes, counters)


class MainDebug(Main):

    def __init__(self):
        super().__init__()

    def parse_goal_frame(self, frame, edgeFrame, bboxes):
        frame, edgeFrame, bboxes = super().parse_goal_frame(frame, edgeFrame, bboxes)
        valid_labels = ['red_upper_power_port', 'blue_upper_power_port']

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

    def parse_intake_frame(self, frame,  bboxes, counters):
        frame, bboxes, counters = super().parse_intake_frame(frame, bboxes, counters)

        for i, bbox in enumerate(bboxes):
            angle_offset = bbox['angle_offset'] if 'angle_offset' in bbox else 0

            frame_color = (0, 255, 0) if i == 0 else (0, 150, 150)

            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), frame_color, 2)
            cv2.putText(frame, "x: {}".format(round(bbox['x_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "y: {}".format(round(bbox['y_mid'], 2)), (bbox['x_min'], bbox['y_min'] + 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "angle: {}".format(round(angle_offset, 3)), (bbox['x_min'], bbox['y_min'] + 70),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "size: {}".format(round(bbox['size'], 3)), (bbox['x_min'], bbox['y_min'] + 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
            cv2.putText(frame, "conf: {}".format(round(bbox['confidence'], 2)), (bbox['x_min'], bbox['y_min'] + 110),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

        # cv2.imshow("OAK-1 Intake Edge", edgeFrame)
        cv2.imshow("OAK-1 Intake", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    log.info("Starting goal-depth-detection-host")
    if args.debug:
        MainDebug().run()
    else:
        Main().run()
