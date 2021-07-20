import cv2
import depthai as dai


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define source camera
rgb_camera = pipeline.createColorCamera()
rgb_camera.setPreviewSize(640, 480)
rgb_camera.setInterleaved(False)
rgb_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create output
rgb_xout = pipeline.createXLinkOut()
rgb_xout.setStreamName("RGB")
rgb_camera.preview.link(rgb_xout.input)

# Connect to the device
with dai.Device() as device:

    device.startPipeline(pipeline)
    rgb_queue = device.getOutputQueue(name="RGB", maxSize=1, blocking=False)

    while True:
        rgb_in = rgb_queue.get()
        frame = rgb_in.getCvFrame()

        cv2.imshow('FRAME', frame)

        cropped = frame[0:480, 80:560]

        cv2.imshow('CROP', cropped)

        if cv2.waitKey(1) == ord('q'):
            break