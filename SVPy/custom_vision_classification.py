import cv2
import numpy as np
from openvino.inference_engine import IECore
from aux_classes import Classification

class ImageClassification:

    def __init__(self, inference_engine, model_path, prob_threshold, max_detections):
        model_xml = model_path + 'model.xml'
        model_bin = model_path + 'model.bin'
        model_labels = model_path + 'labels.txt'

        self.prob_threshold = prob_threshold
        self.max_detections = max_detections

        with open(model_labels, 'r') as io:
            self.labels = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in io]

        network = inference_engine.read_network(model=model_xml, weights=model_bin)

        network.batch_size = 1

        self.input_blob = next(iter(network.inputs))
        self.output_blob = next(iter(network.outputs))

        n, c, h, w = network.inputs[self.input_blob].shape

        self.exec_network = inference_engine.load_network(network=network, device_name='AUTO')
        # self.exec_network = inference_engine.load_network(network=network, device_name='MYRIAD')
        # self.exec_network = inference_engine.load_network(network=network, device_name='MULTI:MYRIAD.1.1.1-ma2480')
        # self.exec_network = inference_engine.load_network(network=network, device_name='MULTI:MYRIAD.1.1.2-ma2480')

        self.images = np.ndarray(shape=(n, c, h, w))

    def Infer(self, image):
        self.images[0] = self._preprocess(image)

        result = self.exec_network.infer(inputs={self.input_blob: self.images})

        outputs = result[self.output_blob]

        return self._postprocess(outputs)

    def _preprocess(self, image):
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        input_image = resized_image.transpose((2, 0, 1))

        return input_image

    def _postprocess(self, outputs):
        predictions = list()
        
        for probs in outputs:
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-self.max_detections:][::-1]
            for id in top_ind:
                if probs[id] > self.prob_threshold:
                    prob = probs[id] * 100
                    prediction = Classification(self.labels[id], prob)
                    predictions.append(prediction)

        return predictions