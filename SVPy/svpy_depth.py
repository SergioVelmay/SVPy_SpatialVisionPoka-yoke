import argparse

parser = argparse.ArgumentParser()
parser.add_argument("code", type=int, help="set regions and display markers for a part")

args = parser.parse_args()

part_code = int(args.code)

print('SVPy Part Code:', part_code)

import depthai as dai
import cv2

pipeline = dai.Pipeline()

w = 250
h = 200

NAME_COLOR = 'Color'
camColor = pipeline.createColorCamera()
camColor.setPreviewSize(3840, 2160)
camColor.setBoardSocket(dai.CameraBoardSocket.RGB)
camColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camColor.setInterleaved(False)
camColor.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camColorLink = pipeline.createXLinkOut()
camColorLink.setStreamName(NAME_COLOR)
camColor.preview.link(camColorLink.input)

NAME_LEFT = 'Left'
camLeft = pipeline.createMonoCamera()
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
camLeftManip = pipeline.createImageManip()
camLeft_x = 619
camLeft_y = 255
camLeft_left_top_x = camLeft_x / 1280
camLeft_left_top_y = camLeft_y / 800
camLeft_right_bot_x = camLeft_left_top_x + (w / 1280)
camLeft_right_bot_y = camLeft_left_top_y + (h / 800)
camLeftTopLeft = dai.Point2f(camLeft_left_top_x, camLeft_left_top_y)
camLeftBotRight = dai.Point2f(camLeft_right_bot_x, camLeft_right_bot_y)
camLeftManip.initialConfig.setCropRect(
    camLeftTopLeft.x, camLeftTopLeft.y, camLeftBotRight.x, camLeftBotRight.y)
camLeftManip.setMaxOutputFrameSize(
    camLeft.getResolutionWidth()*camLeft.getResolutionHeight()*3)
camLeft.out.link(camLeftManip.inputImage)
camLeftLink = pipeline.createXLinkOut()
camLeftLink.setStreamName(NAME_LEFT)
camLeftManip.out.link(camLeftLink.input)

NAME_RIGHT = 'Right'
camRight = pipeline.createMonoCamera()
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
camRightManip = pipeline.createImageManip()
camRight_x = 503
camRight_y = 255
camRight_left_top_x = camRight_x / 1280
camRight_left_top_y = camRight_y / 800
camRight_right_bot_x = camRight_left_top_x + (w / 1280)
camRight_right_bot_y = camRight_left_top_y + (h / 800)
camRightTopLeft = dai.Point2f(camRight_left_top_x, camRight_left_top_y)
camRightBotRight = dai.Point2f(camRight_right_bot_x, camRight_right_bot_y)
camRightManip.initialConfig.setCropRect(
    camRightTopLeft.x, camRightTopLeft.y, camRightBotRight.x, camRightBotRight.y)
camRightManip.setMaxOutputFrameSize(
    camRight.getResolutionWidth()*camRight.getResolutionHeight()*3)
camRight.out.link(camRightManip.inputImage)
camRightLink = pipeline.createXLinkOut()
camRightLink.setStreamName(NAME_RIGHT)
camRightManip.out.link(camRightLink.input)

NAME_STEREO = 'Stereo'
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(255)
stereo.setOutputDepth(False)
stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereoLink = pipeline.createXLinkOut()
stereoLink.setStreamName(NAME_STEREO)
stereo.disparity.link(stereoLink.input)

NAME_DEPTH = 'Depth'
NAME_DATA = 'Data'
NAME_SPACE = 'Space'
space = pipeline.createSpatialLocationCalculator()
depthLink = pipeline.createXLinkOut()
dataLink = pipeline.createXLinkOut()
spaceLink = pipeline.createXLinkIn()
depthLink.setStreamName(NAME_DEPTH)
dataLink.setStreamName(NAME_DATA)
spaceLink.setStreamName(NAME_SPACE)
space.passthroughDepth.link(depthLink.input)
stereo.depth.link(space.inputDepth)
space.setWaitForConfigInput(False)

def calc_x(x):
    return x / 1280

def calc_y(y):
    return y / 800

def calc_rect(x, y):
    return dai.Rect(
        dai.Point2f(calc_x(x-4), calc_y(y-4)), 
        dai.Point2f(calc_x(x+4), calc_y(y+4)))

regions = {
    0: [(566, 352), (595, 352), (624, 352), (653, 352), (682, 352)],
    1: [(560, 311), (683, 311), (560, 393), (683, 393)],
    2: [(574, 352), (668, 352)],
    3: [(619, 321), (619, 352), (619, 390)],
    4: [(599, 351), (646, 351)],
    5: [(602, 337), (635, 337), (602, 369), (635, 369)],
    6: [(620, 349)],
    7: [(602, 335), (635, 335), (602, 367), (635, 367)],
    8: [(561, 351), (679, 351)]
}

markers = {
    0: [(256, 367), (357, 367), (458, 367), (559, 367), (660, 367)],
    1: [(226, 220), (670, 220), (226, 514), (670, 514)],
    2: [(286, 364), (612, 364)],
    3: [(450, 268), (450, 364), (450, 499)],
    4: [(373, 360), (535, 360)],
    5: [(402, 298), (511, 298), (402, 411), (511, 411)],
    6: [(460, 350)],
    7: [(392, 298), (501, 298), (392, 411), (501, 411)],
    8: [(250, 355), (650, 355)]
}

roi_regions = regions[part_code]

for roi_region in roi_regions:
    roi = calc_rect(roi_region)
    location = dai.SpatialLocationCalculatorConfigData()
    location.depthThresholds.lowerThreshold = 300
    location.depthThresholds.upperThreshold = 1800
    location.roi = roi
    space.initialConfig.addROI(location)

space.out.link(dataLink.input)
spaceLink.out.link(space.inputConfig)

with dai.Device(pipeline) as device:

    queueColor = device.getOutputQueue(NAME_COLOR, maxSize=4, blocking=False)
    queueLeft = device.getOutputQueue(NAME_LEFT, maxSize=4, blocking=False)
    queueRight = device.getOutputQueue(NAME_RIGHT, maxSize=4, blocking=False)
    queueStereo = device.getOutputQueue(NAME_STEREO, maxSize=4, blocking=False)
    queueDepth = device.getOutputQueue(name=NAME_DEPTH, maxSize=4, blocking=False)
    queueData = device.getOutputQueue(name=NAME_DATA)

    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2

    def displayFrame(name, frame):
        cv2.imshow(name, frame)
    
    def draw_markers(image, markers):
        for mark in markers:
            cv2.drawMarker(croppedColor, mark, color, cv2.MARKER_CROSS, 36, thick)
            cv2.drawMarker(croppedColor, mark, color, cv2.MARKER_SQUARE, 18, 1)
        return image

    while True:

        inColor = queueColor.get()
        frameColor = inColor.getCvFrame()
        inLeft = queueLeft.get()
        frameLeft = inLeft.getCvFrame()
        inRight = queueRight.get()
        frameRight = inRight.getCvFrame()
        inDepth = queueDepth.get()
        inData = queueData.get()
        frameDepth = inDepth.getFrame()

        rgbLeft = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2RGB)
        scaledLeft = resized = cv2.resize(rgbLeft, (300, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(scaledLeft, "Left Camera", (12, 27), font, scale, color, thick)

        rgbRight = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2RGB)
        scaledRight = resized = cv2.resize(rgbRight, (300, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(scaledRight, "Right Camera", (12, 27), font, scale, color, thick)

        depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        croppedColor = frameColor[522:1242, 1614:2514]
        cv2.putText(croppedColor, "Color Camera", (12, 27), font, scale, color, thick)
        cv2.putText(croppedColor, "Height mm", (780, 27), font, scale, color, thick)

        color_markers = draw_markers(croppedColor, markers[args.code])

        spatialData = inData.getSpatialLocations()

        for index, depthData in enumerate(spatialData):
            z_value = depthData.spatialCoordinates.z
            text_coords = (markers[part_code][index][0] + 8, markers[part_code][index][1] + 24)
            if z_value:
                text = f"{int(z_value / 2)}"
            else:
                text = ""
            cv2.putText(croppedColor, text, text_coords, font, scale, color, thick)
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            x = int(roi.topLeft().x + ((roi.bottomRight().x - roi.topLeft().x) / 2))
            y = int(roi.topLeft().y + ((roi.bottomRight().y - roi.topLeft().y) / 2))
            cv2.drawMarker(depthFrameColor, (x, y), color, cv2.MARKER_SQUARE, 9, 1)

        depthFrameCropped = depthFrameColor[251:451, 494:744]
        
        depthFrameScaled = cv2.resize(depthFrameCropped, (300, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(depthFrameScaled, "Spatial Depth", (12, 27), font, scale, color, thick)

        combo_vertical = cv2.vconcat([scaledLeft, depthFrameScaled, scaledRight])

        combo = cv2.hconcat([combo_vertical, croppedColor])
        displayFrame('SVPy | Spatial Vision Poka-yoke', combo)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break