import depthai as dai
from threading import Thread

class Camera:
    def __init__(self) -> None:
        self.NAME = 'RGB'
        self.Pipeline = dai.Pipeline()
        self.Create()
        self.Frame = None
        Thread(target=self.Capture, daemon=True).start()

    def Create(self):
        self.RGB_Camera = self.Pipeline.createColorCamera()
        self.RGB_Camera.setPreviewSize(640, 480)
        self.RGB_Camera.setInterleaved(False)
        self.RGB_Camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.RGB_XLinkOut = self.Pipeline.createXLinkOut()
        self.RGB_XLinkOut.setStreamName(self.NAME)
        self.RGB_Camera.preview.link(self.RGB_XLinkOut.input)

    def Capture(self):
        with dai.Device() as device:
            device.startPipeline(self.Pipeline)
            self.RGB_Queue = device.getOutputQueue(name=self.NAME, maxSize=1, blocking=False)
            while True:
                self.RGB_Data = self.RGB_Queue.get()
                self.Frame = self.RGB_Data.getCvFrame()
