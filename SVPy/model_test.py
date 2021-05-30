#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

nnPathDefault = str((Path(__file__).parent / Path('models/part_count_detection/openvino/model.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

if not Path(nnPathDefault).exists():
    import sys
    raise FileNotFoundError(f'Required file not found.')

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
rgb_camera = pipeline.createColorCamera()
rgb_camera.setPreviewSize(416, 416)
rgb_camera.setInterleaved(False)
rgb_camera.setFps(30)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(1)
nn.setNumPoolFrames(1)
nn.input.setBlocking(False)
rgb_camera.preview.link(nn.input)

# Create outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
if args.sync:
    nn.passthrough.link(xoutRgb.input)
else:
    rgb_camera.preview.link(xoutRgb.input)

nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Label texts
labelMap = [       
    'Part.Hole',
    'Part5.False.A',
    'Part5.False.B',
    'Part5.True',
    'Part6.False.A',
    'Part6.False.B',
    'Part6.True']

# Connect and start the pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    print(1)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    print(2)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    print(3)

    startTime = time.monotonic()
    counter = 0
    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.imshow(name, frame)

    while True:
        print(4)
        if args.sync:
            print(5)
            # Use blocking get() call to catch frame and inference result synced
            inRgb = qRgb.get()
            print(inRgb)
            inDet = qDet.get()
            print(inDet)
            print(6)
        else:
            print(7)
            # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
            inRgb = qRgb.tryGet()
            print(inRgb)
            inDet = qDet.tryGet()
            print(inDet)
            print(8)

        if inRgb is not None:
            print(9)
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
            print(10)

        if inDet is not None:
            print(11)
            print(inDet)
            detections = inDet.detections
            counter += 1
            print(12)

        # If the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
