#!/usr/bin/env python3
import time

import depthai as dai
import uuid

from common.config import *
from pathlib import Path

log = logging.getLogger(__name__)


def create_pipeline(model_name):
    global pipeline
    log.debug("Creating DepthAI pipeline...")

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    detectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
    edgeDetectorRgb = pipeline.createEdgeDetector()
    edgeManip = pipeline.createImageManip()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    videoEncoder = pipeline.createVideoEncoder()

    xoutRgb = pipeline.createXLinkOut()
    rgbControl = pipeline.createXLinkIn()
    # xinRgb = pipeline.createXLinkIn()
    xoutNN = pipeline.createXLinkOut()
    xoutEdge = pipeline.createXLinkOut()
    videoOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    # xinRgb.setStreamName("rgbCfg")
    rgbControl.setStreamName('rgbControl')
    xoutNN.setStreamName("detections")
    xoutEdge.setStreamName("edge")
    videoOut.setStreamName("h265")

    # Properties
    camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
    # camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)
    camRgb.initialControl.setManualExposure(100000, 300)

    videoEncoder.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)

    edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())
    edgeManip.initialConfig.setResize(NN_IMG_SIZE, NN_IMG_SIZE)

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
    camRgb.preview.link(detectionNetwork.input)
    # detectionNetwork.passthrough.link(xoutRgb.input)
    camRgb.preview.link(xoutRgb.input)
    rgbControl.out.link(camRgb.inputControl)
    # xinRgb.out.link(camRgb.inputConfig)
    detectionNetwork.out.link(xoutNN.input)

    camRgb.video.link(edgeDetectorRgb.inputImage)
    edgeDetectorRgb.outputImage.link(edgeManip.inputImage)
    edgeManip.out.link(xoutEdge.input)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(detectionNetwork.inputDepth)

    camRgb.video.link(videoEncoder.input)
    videoEncoder.bitstream.link(videoOut.input)

    log.debug("Pipeline created.")

    return pipeline, labels


def capture(device_info):
    filename = '{}.h265'.format(time.strftime("%Y_%m_%d-%H_%M_%S"))
    with dai.Device(pipeline, device_info) as device, open(filename, 'wb') as videoFile:
    # with dai.Device(pipeline, device_info) as device:
        log.warning("VIDEO ENCODING ENABLED")
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        edgeQueue = device.getOutputQueue("edge", 8, False)
        qRgbEnc = device.getOutputQueue('h265', maxSize=30, blocking=True)

        # configQueue = device.getInputQueue('rgbCfg')

        while True:
            frame = previewQueue.get().getCvFrame()
            inDet = detectionNNQueue.tryGet()
            # edgeFrame = edgeRgbQueue.get().getFrame()
            edgeFrame = edgeQueue.get().getCvFrame()

            detections = []
            if inDet is not None:
                detections = inDet.detections

            while qRgbEnc.has():
                qRgbEnc.get().getData().tofile(videoFile)

            bboxes = []
            height = edgeFrame.shape[0]
            width  = edgeFrame.shape[1]
            for detection in detections:
                bboxes.append({
                    'id': uuid.uuid4(),
                    'label': detection.label,
                    'confidence': detection.confidence,
                    'x_min': int(detection.xmin * width),
                    'x_mid': int(((detection.xmax - detection.xmin) / 2 + detection.xmin) * width),
                    'x_max': int(detection.xmax * width),
                    'y_min': int(detection.ymin * height),
                    'y_mid': int(((detection.ymax - detection.ymin) / 2 + detection.ymin) * height),
                    'y_max': int(detection.ymax * height),
                    'depth_x': detection.spatialCoordinates.x / 1000,
                    'depth_y': detection.spatialCoordinates.y / 1000,
                    'depth_z': detection.spatialCoordinates.z / 1000,
                })

            yield frame, edgeFrame, bboxes


def del_pipeline():
    del pipeline
