import cv2
import depthai as dai
import numpy as np
from pathlib import Path

from blob_classes_prediction import Classification


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
nnet_path = str((Path(__file__).parent / Path('models/multiclass_classification/openvino/model.blob')).resolve().absolute())
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

PROB_THRESHOLD_CLASS = 0.5
MAX_DETECTIONS_CLASS = 1

LABELS_CLASS = [
    'Step7.false.A',
    'Step7.false.B',
    'Step7.start',
    'Step7.true']

def postprocess_classification(outputs):
    predictions = list()
    
    for probs in outputs:
        predictions = list()
        probs = np.array(outputs)
        top_ids = np.argsort(outputs)[::-1]
        for id in top_ids:
            if len(predictions) < MAX_DETECTIONS_CLASS:
                prob = probs[id]
                if prob > PROB_THRESHOLD_CLASS:
                    prob = prob * 100
                    prediction = Classification(LABELS_CLASS[id], prob)
                    predictions.append(prediction)
        return predictions

def displayFrame(frame, detections):
    for detection in detections:
        cv2.putText(frame, f"{detection.Label} {int(detection.Score)}%", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, thickness=2)
    cv2.imshow('SVPy | Spatial Vision Poka-yoke', frame)

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
        resized = cv2.resize(cropped, (224, 224))
        input = resized.transpose(2, 0, 1)

        input_data = dai.NNData()
        input_data.setLayer("Placeholder", input)
        frame_queue.send(input_data)


        nnet_path = str((Path(__file__).parent / Path('models/multiclass_classification/openvino/model.blob')).resolve().absolute())
        nnet.setBlobPath(nnet_path)

        nnet_in = nnet_quque.get()

        if nnet_in is not None:
            layer_names = nnet_in.getAllLayerNames()
            layer_float = nnet_in.getLayerFp16(layer_names[0])
            detections = postprocess_classification(layer_float)
            displayFrame(frame, detections)

        if cv2.waitKey(1) == ord('q'):
            break