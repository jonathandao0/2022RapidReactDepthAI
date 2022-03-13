#!/usr/bin/env python3
import time

import cv2
import depthai as dai
import uuid

import imutils
import numpy as np

from common.config import *
from pathlib import Path

log = logging.getLogger(__name__)

ENABLE_RECORDING = False


def create_pipeline(model_name):
    global pipeline
    log.debug("Creating DepthAI pipeline...")

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    detectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
    resizeManip = pipeline.createImageManip()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    # xoutRgb = pipeline.createXLinkOut()
    # xoutRgbPreview = pipeline.createXLinkOut()
    # xinRgb = pipeline.createXLinkIn()
    xoutNN = pipeline.createXLinkOut()
    xoutMeta = pipeline.createXLinkOut()
    xoutResize = pipeline.createXLinkOut()

    # xoutRgb.setStreamName("rgb")
    # xoutRgbPreview.setStreamName("rgb_preview")
    # xinRgb.setStreamName("rgbCfg")
    xoutNN.setStreamName("detections")
    xoutResize.setStreamName("resize")
    xoutMeta.setStreamName("metadata")
    xoutMeta.setMetadataOnly(True)

    # Properties
    camRgb.setPreviewSize(416, 234)
    # camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)
    camRgb.setIspScale(13, 60)
    # Exposure time (microseconds), ISO Sensitivity (100-1600)
    # camRgb.initialControl.setManualExposure(20000, 300)
    # # Temperature in Kelvins (1000-12000)
    # camRgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.OFF)
    # camRgb.initialControl.setManualWhiteBalance(4000)
    # camRgb.initialControl.setBrightness(-3)

    resizeManip.setMaxOutputFrameSize(519168)   # 416*416*3
    resizeManip.initialConfig.setResizeThumbnail(NN_IMG_SIZE, NN_IMG_SIZE)
    # resizeManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Setting node configs
    # stereo.setOutputDepth(out_depth)
    # stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(255)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)  # KERNEL_7x7 default
    # stereo.setLeftRightCheck(lrcheck)
    # stereo.setExtendedDisparity(extended)
    # stereo.setSubpixel(subpixel)

    model_dir = Path(__file__).parent.parent / Path(f"resources/nn/") / model_name
    blob_path = model_dir / Path(model_name).with_suffix(f".blob")

    config_path = model_dir / Path(model_name).with_suffix(f".json")
    nn_config = NNConfig(config_path)
    labels = nn_config.labels

    detectionNetwork.setBlobPath(str(blob_path))
    detectionNetwork.setConfidenceThreshold(nn_config.confidence)
    # detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(nn_config.metadata["classes"])
    detectionNetwork.setCoordinateSize(nn_config.metadata["coordinates"])
    detectionNetwork.setAnchors(nn_config.metadata["anchors"])
    detectionNetwork.setAnchorMasks(nn_config.metadata["anchor_masks"])
    detectionNetwork.setIouThreshold(nn_config.metadata["iou_threshold"])
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking
    # camRgb.preview.link(detectionNetwork.input)
    # detectionNetwork.passthrough.link(xoutRgb.input)
    # camRgb.preview.link(xoutRgbPreview.input)
    # camRgb.video.link(xoutRgb.input)
    # xinRgb.out.link(camRgb.inputConfig)

    camRgb.preview.link(resizeManip.inputImage)
    resizeManip.out.link(detectionNetwork.input)
    resizeManip.out.link(xoutResize.input)

    detectionNetwork.out.link(xoutNN.input)
    detectionNetwork.passthrough.link(xoutMeta.input)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(detectionNetwork.inputDepth)

    if ENABLE_RECORDING:
        videoEncoder = pipeline.createVideoEncoder()
        videoEncoder.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
        videoEncoder.setLossless(True)
        videoOut = pipeline.create(dai.node.XLinkOut)
        videoOut.setStreamName("h265")
        camRgb.video.link(videoEncoder.input)
        videoEncoder.bitstream.link(videoOut.input)

    log.debug("Pipeline created.")

    return pipeline, labels


def capture(device_info):
    # filePath = 'empty_file'
    # if ENABLE_RECORDING:
    #     log.warning("VIDEO ENCODING ENABLED")
    #     filePath = 'recordings/goal/{}.h265'.format(time.strftime("%Y_%m_%d-%H_%M_%S"))

    # with dai.Device(pipeline, device_info) as device, open(filePath, 'wb') as videoFile:
    with dai.Device(pipeline, device_info) as device:
        # rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        # previewQueue = device.getOutputQueue(name="rgb_preview", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
        resizeQueue = device.getOutputQueue("resize", 1, False)
        metadataQueue = device.getOutputQueue("metadata", 1, False)

        # if ENABLE_RECORDING:
        #     qRgbEnc = device.getOutputQueue('h265', maxSize=30, blocking=False)

        # configQueue = device.getInputQueue('rgbCfg')

        while True:
            # frame = rgbQueue.get().getCvFrame()
            # frame = previewQueue.get().getCvFrame()
            inDet = detectionNNQueue.tryGet()
            timestamp = metadataQueue.get().getTimestampDevice()
            # edgeFrame = edgeRgbQueue.get().getFrame()
            frame = resizeQueue.get().getCvFrame()

            detections = []
            if inDet is not None:
                detections = inDet.detections

            # if ENABLE_RECORDING:
            #     while qRgbEnc.has():
            #         qRgbEnc.get().getData().tofile(videoFile)

            # frame = frame[91:324, 0:NN_IMG_SIZE]
            bboxes = []
            x_offset = 0
            y_offset = 0
            for detection in detections:
                (xmin, ymin, xmax, ymax) = normalize_detections(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                bboxes.append({
                    'id': uuid.uuid4(),
                    'label': detection.label,
                    'confidence': detection.confidence,
                    'x_min': xmin + x_offset,
                    'x_mid': int(((xmax - xmin) / 2 + xmin)) + x_offset,
                    'x_max': xmax + x_offset,
                    'y_min': ymin,
                    'y_mid': int(((ymax - ymin) / 2 + ymin)),
                    'y_max': ymax,
                    'size': (ymax - ymin) * (xmax - xmin),
                    'depth_x': detection.spatialCoordinates.x / 1000,
                    'depth_y': detection.spatialCoordinates.y / 1000,
                    'depth_z': detection.spatialCoordinates.z / 1000,
                })

            metadata = {
                'timestamp': timestamp
            }

            yield frame, bboxes, metadata


def normalize_detections(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def del_pipeline():
    del pipeline
