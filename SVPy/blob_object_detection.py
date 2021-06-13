import cv2
import depthai as dai
import numpy as np
from pathlib import Path

from SVPy.classes_prediction import Detection


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

# Define a neural network that will make predictions based on the source frames
nnet = pipeline.createNeuralNetwork()
nnet_path = str((Path(__file__).parent / Path('models/part_count_detection/openvino/model.blob')).resolve().absolute())
nnet.setBlobPath(nnet_path)
nnet.setNumInferenceThreads(2)
nnet.setNumPoolFrames(2)
nnet.input.setBlocking(False)

# Input stream
frame_in = pipeline.createXLinkIn()
frame_in.setStreamName("FRAME")
frame_in.out.link(nnet.input)

# Create outputs
nnet_xout = pipeline.createXLinkOut()
nnet_xout.setStreamName("NNET")
nnet.out.link(nnet_xout.input)

PROB_THRESHOLD = 0.2
MAX_DETECTIONS = 5

ANCHOR_BOXES = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
IOU_THRESHOLD = 0.5

LABELS = [       
    'Part.Hole',
    'Part5.False.A',
    'Part5.False.B',
    'Part5.True',
    'Part6.False.A',
    'Part6.False.B',
    'Part6.True']

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
        label = LABELS[selected_classes[i]]
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

def displayFrame(frame, detections):
    for id, detection in enumerate(detections):
        cv2.rectangle(frame, 
        (int(detection.Box.Left * 480) + 80, int(detection.Box.Top * 480)), 
        (int((detection.Box.Left + detection.Box.Width) * 480) + 80, int((detection.Box.Top + detection.Box.Height) * 480)), 
        (255, 0, 0), 2)
        cv2.putText(frame, f"{detection.Label} {int(detection.Probability)}%", (int(detection.Box.Left * 480) + 80 + 10, int(detection.Box.Top * 480) + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.imshow('COMBO', frame)

# Connect to the device
with dai.Device() as device:

    device.startPipeline(pipeline)
    rgb_queue = device.getOutputQueue(name="RGB", maxSize=1, blocking=False)
    frame_queue = device.getInputQueue(name="FRAME", maxSize=1, blocking=True)
    nnet_quque = device.getOutputQueue(name="NNET", maxSize=1, blocking=True)
    frame = None

    while True:
        rgb_in = rgb_queue.get()
        frame = rgb_in.getCvFrame()

        cropped = frame[0:480, 80:560]
        resized = cv2.resize(cropped, (416, 416))
        input = resized.transpose(2, 0, 1)

        input_data = dai.NNData()
        input_data.setLayer("Placeholder", input)
        frame_queue.send(input_data)


        nnet_path = str((Path(__file__).parent / Path('models/part_count_detection/openvino/model.blob')).resolve().absolute())
        nnet.setBlobPath(nnet_path)

        nnet_in = nnet_quque.get()

        if nnet_in is not None:
            layer_names = nnet_in.getAllLayerNames()
            layer_float = nnet_in.getLayerFp16(layer_names[0])
            detections = postprocess(layer_float)
            displayFrame(frame, detections)

        if cv2.waitKey(1) == ord('q'):
            break