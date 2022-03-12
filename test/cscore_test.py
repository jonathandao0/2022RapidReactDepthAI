import CsCoreStream

mjpegServer = CsCoreStream.MjpegServer("httpserver", 5802)
mjpegServer.setSource(cam)

