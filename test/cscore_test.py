import CSCoreStream

mjpegServer = CSCoreStream.MjpegServer("httpserver", 5802)
mjpegServer.setSource(cam)

