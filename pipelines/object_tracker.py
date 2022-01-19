#!/usr/bin/env python3

import depthai as dai
import uuid

from common.config import *
from pathlib import Path

log = logging.getLogger(__name__)


def create_pipeline(model_name):
    global pipeline
    log.info("Creating DepthAI pipeline...")

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    detectionNetwork = pipeline.createYoloDetectionNetwork()
    # edgeDetectorRgb = pipeline.createEdgeDetector()
    # edgeManip = pipeline.createImageManip()
    objectTracker = pipeline.createObjectTracker()
    script = pipeline.create(dai.node.Script)

    xoutRgb = pipeline.createXLinkOut()
    # rgbControl = pipeline.createXLinkIn()
    # xinRgb = pipeline.createXLinkIn()
    xoutNN = pipeline.createXLinkOut()
    # xoutEdgeRgb = pipeline.createXLinkOut()
    # xoutEdge = pipeline.createXLinkOut()
    # xinEdgeCfg = pipeline.createXLinkIn()
    trackerOut = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    # xinRgb.setStreamName("rgbCfg")
    # rgbControl.setStreamName('rgbControl')
    xoutNN.setStreamName("detections")
    # xoutEdgeRgb.setStreamName("edgeRgb")
    # xinEdgeCfg.setStreamName("edgeCfg")
    # xoutEdge.setStreamName("edge")
    trackerOut.setStreamName("out")

    # Properties
    camRgb.setPreviewSize(NN_IMG_SIZE, NN_IMG_SIZE)
    # camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)

    # edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())
    # edgeManip.initialConfig.setResize(NN_IMG_SIZE, NN_IMG_SIZE)

    objectTracker.setDetectionLabelsToTrack([2, 3])
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

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
    # rgbControl.out.link(camRgb.inputControl)
    # xinRgb.out.link(camRgb.inputConfig)
    detectionNetwork.out.link(xoutNN.input)
    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)

    objectTracker.out.link(script.inputs['tracklets'])
    script.outputs['out'].link(trackerOut.input)

    with open("../pipelines/object_counter_script.py", "r") as f:
        s = f.read()
        s = s.replace("LABELS = []", "LABELS = [ 'upper_hub', 'lower_hub', 'red_cargo', 'blue_cargo' ]")
        s = s.replace("COUNTER = {}", "COUNTER = { 'upper_hub': 0, 'lower_hub': 0, 'red_cargo': 0, 'blue_cargo': 0 }")
        s = s.replace("THRESH_DIST_DELTA", "0.25")
        script.setScript(s)

    # camRgb.video.link(edgeDetectorRgb.inputImage)
    # edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)
    # edgeDetectorRgb.outputImage.link(edgeManip.inputImage)
    # edgeManip.out.link(xoutEdge.input)
    # xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

    log.info("Pipeline created.")

    return pipeline, labels


def capture(device_info):
    with dai.Device(pipeline, device_info) as device:
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        # # edgeRgbQueue = device.getOutputQueue("edgeRgb", 8, False)
        # edgeQueue = device.getOutputQueue("edge", 8, False)
        # edgeCfgQueue = device.getInputQueue("edgeCfg")
        outQueue = device.getOutputQueue("out")

        # controlQueue = device.getInputQueue('rgbControl')
        # configQueue = device.getInputQueue('rgbCfg')

        counters = {
            'upper_hub': 0,
            'lower_hub': 0,
            'red_cargo': 0,
            'blue_cargo': 0
        }
        while True:
            # cfg = dai.CameraControl()
            # cfg.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            # cfg.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.OFF)
            # cfg.setAutoExposureLock(True)
            # # cfg.setAutoExposureCompensation(-6)
            # configQueue.send(cfg)

            frame = previewQueue.get().getCvFrame()
            inDet = detectionNNQueue.tryGet()
            # edgeFrame = edgeRgbQueue.get().getFrame()
            # edgeFrame = edgeQueue.get().getCvFrame()

            detections = []
            if inDet is not None:
                detections = inDet.detections

            if outQueue.has():
                jsonText = str(outQueue.get().getData(), 'utf-8')
                counters = json.loads(jsonText)

            bboxes = []
            height = frame.shape[0]
            width  = frame.shape[1]
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
                    'size': ((detection.ymax - detection.ymin) * height) * ((detection.xmax - detection.xmin) * width)
                })

            yield frame, bboxes, counters


def del_pipeline():
    del pipeline
