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
nn.setNumInferenceThreads(2)
nn.setNumPoolFrames(2)
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
labels = [       
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
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

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
            bbox = frameNorm(frame, (detection.Box.Left, detection.Box.Top, detection.Box.Width, detection.Box.Height))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, detection.Label, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.Probability)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.imshow(name, frame)









    class Classification:

        def __init__(self, label, probability):
            self.Label = label
            self.Probability = probability

        def __str__(self):
            words = str.upper(self.Label).split('.')
            if (self.Label.startswith('Step') or self.Label.startswith('Part')):
                words[0] = words[0][:4] + ' #' + words[0][4:]
            return ' - '.join(tuple(words)) + '   ( {:.1f}'.format(self.Probability) + '% )'

    class Boundary:

        def __init__(self, x, y, w, h):
            self.Left = x
            self.Top = y
            self.Width = w
            self.Height = h

        def __str__(self):
            return 'x:{:.2f} y:{:.2f} w:{:.2f} h:{:.2f}'.format(
                self.Left, self.Top, self.Width, self.Height)

    class Detection(Classification):

        def __init__(self, label, probability, x, y, w, h):
            Classification.__init__(self, label, probability)
            self.Box = Boundary(x, y, w, h)

    PROB_THRESHOLD = 0.2
    MAX_DETECTIONS = 5

    ANCHOR_BOXES = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
    IOU_THRESHOLD = 0.5

    def postprocess(outputs):
        outputs = np.array(outputs)

        outputs = outputs.reshape([1,60,13,13])

        outputs = np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

        boxes, class_probs = extract_bounding_boxes(outputs)

        max_probs = np.amax(class_probs, axis=1)
        index, = np.where(max_probs > PROB_THRESHOLD)
        index = index[(-max_probs[index]).argsort()]

        selected_boxes, selected_classes, selected_probs = non_maximum_suppression(
            boxes[index], class_probs[index])

        predictions = list()

        for i in range(len(selected_boxes)):
            label = labels[selected_classes[i]]
            probability = selected_probs[i] * 100
            left = round(float(selected_boxes[i][0]), 8)
            top = round(float(selected_boxes[i][1]), 8)
            width = round(float(selected_boxes[i][2]), 8)
            height = round(float(selected_boxes[i][3]), 3)
            prediction = Detection(label, probability, left, top, width, height)
            predictions.append(prediction)

        return predictions

    def extract_bounding_boxes(output):
        num_anchor = ANCHOR_BOXES.shape[0]
        height, width, channels = output.shape
        num_class = int(channels / num_anchor) - 5

        outputs = output.reshape((height, width, num_anchor, -1))

        x = (logistic(outputs[..., 0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (logistic(outputs[..., 1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * ANCHOR_BOXES[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * ANCHOR_BOXES[:, 1][np.newaxis, np.newaxis, :] / height

        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, w, h), axis=-1).reshape(-1, 4)

        objectness = logistic(outputs[..., 4])

        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness[..., np.newaxis]
        class_probs = class_probs.reshape(-1, num_class)

        return (boxes, class_probs)

    def logistic(x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def non_maximum_suppression(boxes, class_probs):
        max_detections = min(MAX_DETECTIONS, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)

        areas = boxes[:, 2] * boxes[:, 3]

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            i = np.argmax(max_probs)
            if max_probs[i] < PROB_THRESHOLD:
                break

            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[0] + box[2], other_boxes[:, 0] + other_boxes[:, 2])
            y2 = np.minimum(box[1] + box[3], other_boxes[:, 1] + other_boxes[:, 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            overlapping_indices = other_indices[np.where(iou > IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            class_probs[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(class_probs[overlapping_indices], axis=1)
            max_classes[overlapping_indices] = np.argmax(class_probs[overlapping_indices], axis=1)

        return selected_boxes, selected_classes, selected_probs












    while True:
        if args.sync:
            # Use blocking get() call to catch frame and inference result synced
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        if inDet is not None:
            layer_names = inDet.getAllLayerNames()
            layer_float = inDet.getLayerFp16(layer_names[0])
            detections = postprocess(layer_float)
            counter += 1

        # If the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break


