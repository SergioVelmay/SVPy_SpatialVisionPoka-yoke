import cv2
import depthai as dai
from PIL import ImageTk, Image
from datetime import datetime
import logging
from openvino.inference_engine import IECore
from time import sleep
from gpiozero import LED, Buzzer
import warnings

warnings.filterwarnings("ignore")

from aux_constants import colors_hex, colors_bgr, welcome
from aux_constants import multilabel_labels, multilabel_help, multilabel_error
from aux_constants import detection_labels, detection_help, detection_error
from aux_constants import multiclass_labels, multiclass_help, multiclass_error

number_of_models = 3
number_of_steps = 8
number_of_messages = [
    len(multilabel_help), len(detection_help), len(multiclass_help)]

step_validations = {}
for step in range(number_of_steps):
    step_validations[str(step)] = 0

message_waitings = {}
for model in range(number_of_models):
    for message in range(number_of_messages[model]):
        message_waitings[str(model) + str(message)] = 0

waiting_millis = 3
waiting_frames = 20

min_validations = 20

welcome_message = welcome
welcome_waiting = 0

current_step = 0
current_model = 0
current_message = 0

assembly_completed = False
completed_count = 0

parts_counting = True
part_counts_ok = 0
part5_max = 4
part6_max = 1

oring_tracking = False
part4_detected = False
oring_counts_ok = 0

glove_tracking = False
hand_detected = False
glove_detected = False

operator_tracking = False
operator_detected = False

frame_number = 0

print('[ SVPy ] Spatial Vision Poka-yoke launched')

logging.basicConfig(format='[ %(levelname)s ] New Image Capture | Frame Count %(message)s | %(asctime)s', level=logging.INFO)

from custom_vision_classification import ImageClassification
from custom_vision_detection import ObjectDetection
from aux_classes import Validation

inference_engine = IECore()
print('[ SVPy ] OpenVINO Inference Engine created')

glove_class = ImageClassification(inference_engine, 'models/gloves_classification/openvino/', 0.5, 1)
print('[ SVPy ] Gloves Classification model loaded')
part_count = ObjectDetection(inference_engine, 'models/part_count_detection/openvino/', 0.2, 5)
print('[ SVPy ] Part Count Detection model loaded')
multilabel = ImageClassification(inference_engine, 'models/multilabel_classification/openvino/', 0.5, 3)
print('[ SVPy ] Multilabel Classification model loaded')
detection = ObjectDetection(inference_engine, 'models/objects_detection/openvino/', 0.5, 5)
print('[ SVPy ] Objects Detection model loaded')
multiclass = ImageClassification(inference_engine, 'models/multiclass_classification/openvino/', 0.5, 1)
print('[ SVPy ] Multiclass Classification model loaded')
part_4_det = ObjectDetection(inference_engine, 'models/hidden_part_detection/openvino/', 0.5, 1)
print('[ SVPy ] Hidden Part Detection model loaded')
oring_class = ImageClassification(inference_engine, 'models/oring_classification/openvino/', 0.5, 1)
print('[ SVPy ] O-Ring Classification model loaded')

from aux_camera import Camera

camera = Camera()
print('[ SVPy ] Depth AI RGB Camera device created')

led_green = LED(21)
led_white = LED(16)
led_red = LED(12)
led_yellow = LED(1)
led_blue = LED(8)

leds = [led_green, led_white, led_red, led_yellow, led_blue]

buzzer = Buzzer(24)

def leds_loading():
    print('[ SVPy ] LEDs and Buzzer GPIO Pins added')
    sleep(0.50)
    for led in leds:
        led.on()
        sleep(0.25)
    buzzer.on()
    sleep(0.25)
    buzzer.off()
    for led in leds:
        led.off()
        sleep(0.25)

leds_loading()

from aux_interface import Window

window = Window()
print('[ SVPy ] Graphical User Interface loaded')

from aux_images import color_images, progress_images
from aux_images import assembly_images, validation_images
from aux_images import completed_image, completed_mask
from aux_images import gloves_image, gloves_mask
from aux_images import caution_image, caution_mask

def video_streaming():
    global frame_number
    global welcome_message
    global welcome_waiting
    global parts_counting
    global oring_tracking
    global part4_detected
    global oring_counts_ok
    global min_validations
    global glove_tracking
    global hand_detected
    global glove_detected
    global operator_tracking
    global operator_detected
    global assembly_completed
    global completed_count

    time_total_start = datetime.now().microsecond
    logging.info(str(frame_number).zfill(6))
    time_inference_start = 0
    time_inference_end = 0
    
    image = camera.Frame

    if welcome_waiting > waiting_frames:
        image_crop = image[0:480, 80:560]
        predictions = []
        time_inference_start = datetime.now().microsecond
        if parts_counting:
            predictions = part_count.Infer(image_crop)
        elif oring_tracking:
            predictions = part_4_det.Infer(image_crop)
            if len(predictions) > 0:
                part4_detected = True
            else:
                part4_detected = False
        else:
            if current_model == 0:
                predictions = multilabel.Infer(image_crop)
            elif current_model == 1:
                predictions = detection.Infer(image_crop)
            elif current_model == 2:
                predictions = multiclass.Infer(image_crop)
        if glove_tracking:
            glove_detected = False
            hand_detected = False
            tracking = glove_class.Infer(image_crop)
            if len(tracking) > 0:
                predictions.append(tracking[0])
                if 'Glove' in tracking[0].Label:
                    glove_detected = True
                if 'Hand' in tracking[0].Label:
                    hand_detected = True
        if operator_tracking:
            tracking = glove_class.Infer(image_crop)
            if len(tracking) > 0:
                predictions.append(tracking[0])
                if 'Negative' in tracking[0].Label:
                    operator_detected = False
                else:
                    operator_detected = True
        time_inference_end = datetime.now().microsecond
        if parts_counting:
            validations = process_part_count(predictions)
            image = draw_part_count(image, validations)
            window.assembly.config(image=assembly_images[6])
            window.currently.config(text='Part counting detections:')
        elif oring_tracking:
            detections = process_part_4_det(predictions)
            image = draw_detections(image, detections)
            if hand_detected:
                text = 'SAFETY GLOVES ARE NEEDED TO RESUME ASSEMBLY'
            else:
                text = 'INSERT THE BLACK O-RING ON THE GREY ROTOR PART'
            if part4_detected:
                zoom = detections[0].Box
                x1 = int(round(zoom.Left * 480))
                xw = int(round(zoom.Width * 480))
                x2 = x1 + xw
                y1 = int(round(zoom.Top * 480))
                yh = int(round(zoom.Height * 480))
                y2 = y1 + yh
                image_zoom = image_crop[y1:y2, x1:x2]
                if image_zoom.size != 0:
                    validations = oring_class.Infer(image_zoom)
                    if len(validations) > 0:
                        predictions.append(validations[0])
                        if 'True' in validations[0].Label:
                            color = colors_bgr['yes']
                            if glove_detected:
                                draw_validation('yes')
                                oring_counts_ok = oring_counts_ok + 1
                                if oring_counts_ok > min_validations:
                                    oring_tracking = False
                                    glove_tracking = False
                                    glove_detected = False
                                    hand_detected = False
                        else:
                            color = colors_bgr['no']
                        cv2.putText(image, 'O-Ring', (x1 + 85, y1 + 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,  
                            fontScale=0.9, thickness=2, color=color, bottomLeftOrigin=False)
            window.instruction.config(text=text)
            window.assembly.config(image=assembly_images[4])
            window.currently.config(text='Part 4 + O-Ring detections:')
        else:
            if current_model == 0:
                process_multilabel(predictions)
            elif current_model == 1:
                detections = process_detection(predictions)
                image = draw_detections(image, detections)
            elif current_model == 2:
                process_multiclass(predictions)
            window.assembly.config(image=assembly_images[current_step])
            print_currently(len(predictions))
        if operator_tracking:
            if operator_detected:
                draw_validation('no')
            window.instruction.config(text='CAUTION! THE ASSEMBLY AREA SHOULD BE CLEAR NOW')
            window.assembly.config(image=assembly_images[7])
            window.currently.config(text='Assembly Area detections:')
        print_detections(predictions)
        print_inference_time(time_inference_start, time_inference_end)
    else:
        write_instruction(welcome_message)
        welcome_waiting = welcome_waiting + 1
    if assembly_completed:
        if completed_count < 40:
            completed_count = completed_count + 1
            draw_validation('yes')
            cv2_array = draw_completed(image)
        else:
            cv2_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            assembly_completed = False
            operator_tracking = True
    elif operator_tracking:
        draw_validation('aux')
        if operator_detected:
            draw_validation('no')
            cv2_array = draw_caution(image)
        else:
            cv2_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif hand_detected:
        draw_validation('no')
        cv2_array = draw_gloves(image)
    else:
        cv2_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    pil_image = Image.fromarray(cv2_array)
    image_tk = ImageTk.PhotoImage(image=pil_image)
    window.streaming.image_tk = image_tk
    window.streaming.config(image=image_tk)
    frame_number = frame_number + 1
    print_total_time(time_total_start)
    window.streaming.after(waiting_millis, video_streaming)

def process_multilabel(predictions):
    global current_message
    global oring_tracking
    global glove_tracking
    global current_step
    global step_validations
    message = multilabel_help[current_message]
    checking_value = 'aux'
    detected_labels = []
    for prediction in predictions:
        detected_labels.append(prediction.Label)
    if current_message != 0:
        if current_message == 1 and multilabel_labels[9] in detected_labels:
            checking_value = 'no'
            message = multilabel_error[0]
        elif multilabel_labels[5] in detected_labels:
            checking_value = 'no'
            message = multilabel_error[1]
        elif multilabel_labels[8] in detected_labels:
            checking_value = 'no'
            message = multilabel_error[2]
            if multilabel_labels[11] in detected_labels:
                message = multilabel_error[3]
        elif len(detected_labels) > 0:
            if current_step == 3:
                condition_1 = multilabel_labels[3] in detected_labels
                condition_2 = multilabel_labels[4] in detected_labels
                if condition_1 and condition_2:
                    checking_value = 'yes'
                    update_message()
                    update_progress()
                    # Activate Part 4 + O-Ring Step
                    if step_validations['3'] == min_validations:
                        oring_tracking = True
                        glove_tracking = True
                elif condition_1:
                    checking_value = 'aux'
            elif current_step == 4:
                condition_1 = multilabel_labels[6] in detected_labels
                condition_2 = multilabel_labels[7] in detected_labels
                if condition_1 and condition_2:
                    checking_value = 'yes'
                    update_message()
                    update_progress()
                elif condition_1:
                    checking_value = 'aux'
            else:
                for label in detected_labels:
                    if str(current_step) in label:
                        checking_value = 'yes'
                        update_message()
                        update_progress()
                        break
    draw_validation(checking_value)
    led_routing(checking_value)
    write_instruction(message)
    if current_message == 0:
        update_message()
        window.bar_images[0].config(image=color_images[0])
        window.bar_progress[0].config(image=progress_images['aux'])
    if current_message == len(multilabel_help) - 1:
        update_model()

def process_detection(predictions):
    global current_message
    message = detection_help[current_message]
    checking_value = 'aux'
    selected_predictions = []
    for prediction in predictions:
        if str(current_step) in prediction.Label:
            selected_predictions.append(prediction)
    selected_labels = []
    for prediction in selected_predictions:
        selected_labels.append(prediction.Label)
    if current_message != 0:
        if detection_labels[2] in selected_labels:
            error_predictions = []
            for prediction in selected_predictions:
                if prediction.Label == detection_labels[2]:
                    error_predictions.append(prediction)
                    break
            selected_predictions = error_predictions
            checking_value = 'no'
            message = detection_error[0]
        if current_step == 5:
            part_labels = 0
            for label in selected_labels:
                if label == detection_labels[1]:
                    part_labels = part_labels + 1
            if part_labels == 4:
                checking_value = 'yes'
                update_message()
                update_progress()
        if current_step == 6:
            if detection_labels[4] in selected_labels:
                checking_value = 'yes'
                update_message()
                update_progress()
    draw_validation(checking_value)
    led_routing(checking_value)
    write_instruction(message)
    if current_message == 0:
        update_message()
    if current_message == len(detection_help) - 1:
        update_model()
    return selected_predictions

def process_multiclass(predictions):
    global current_message
    global current_step
    global current_model
    global assembly_completed
    checking_value = 'aux'
    if current_message == len(multiclass_help):
        current_message = len(multiclass_help) - 1
        current_step = 7
        current_model = 2
        assembly_completed = True
    message = multiclass_help[current_message]
    detected_labels = []
    for prediction in predictions:
            detected_labels.append(prediction.Label)
    if current_message != 0 and current_message != len(multiclass_help) - 1:
        if len(detected_labels) > 0:
            if multiclass_labels[2] in detected_labels:
                checking_value = 'no'
                message = multiclass_error[0]
            elif multiclass_labels[3] in detected_labels:
                checking_value = 'no'
                message = multiclass_error[1]
            elif multiclass_labels[1] in detected_labels:
                checking_value = 'yes'
                update_message()
    draw_validation(checking_value)
    led_routing(checking_value)
    write_instruction(message)
    if current_message == 0:
        update_message()
    if current_message == len(multiclass_help) - 1:
        window.bar_images[current_step].config(
            image=color_images[current_step], bg=colors_hex['yes'])
        window.bar_progress[current_step].config(
            image=progress_images['yes'], bg=colors_hex['yes'])
        draw_validation('yes')
        update_message()

def process_part_4_det(predictions):
    checking_value = 'aux'
    selected_predictions = []
    for prediction in predictions:
        selected_predictions.append(prediction)
    draw_validation(checking_value)
    led_routing(checking_value)
    return selected_predictions

def process_part_count(predictions):
    global part5_max
    global part6_max
    part5_count = 0
    part6_count = 0
    checking_value = 'aux'
    if len(predictions) > 0:
        predictions = order_predictions(predictions)
        for prediction in predictions:
            if 'False' in prediction.Label:
                checking_value = 'no'
            else:
                if '5.T' in prediction.Label:
                    part5_count = part5_count + 1
                elif '6.T' in prediction.Label:
                    part6_count = part6_count + 1
        if part5_count > part5_max or part6_count > part6_max:
            checking_value = 'no'
        if part5_count == part5_max and part6_count == part6_max:
            checking_value = 'yes'
    validations = make_validations(predictions)
    draw_validation(checking_value)
    led_routing(checking_value)
    return validations

def order_predictions(predictions):
    predictions.sort(key=lambda x: x.Box.Top)
    return predictions

def make_validations(predictions):
    global min_validations
    global parts_counting
    global part_counts_ok
    global part5_max
    global part6_max
    part5_count = 0
    part6_count = 0
    validations = []
    for prediction in predictions:
        color = colors_bgr['aux']
        thickness = 2
        text = ''
        if 'False' in prediction.Label:
            color = colors_bgr['no']
            thickness = 6
        elif 'True' in prediction.Label:
            color = colors_bgr['yes']
            thickness = 4
            if '5' in prediction.Label:
                part5_count = part5_count + 1
                if part5_count > part5_max:
                    color = colors_bgr['no']
                text = f'{part5_count}/{part5_max}'
            elif '6' in prediction.Label:
                part6_count = part6_count + 1
                if part6_count > part6_max:
                    color = colors_bgr['no']
                text = f'{part6_count}/{part6_max}'
        validation = Validation(prediction.Label, prediction.Score, 
            prediction.Box.Left, prediction.Box.Top, prediction.Box.Width, prediction.Box.Height, 
            color, thickness, text)
        validations.append(validation)
    part5_total = part5_max - part5_count
    part6_total = part6_max - part6_count
    part5_prefix = ''
    part6_prefix = ''
    if part5_total > 0:
        part5_prefix = '+'
    if part6_total > 0:
        part6_prefix = '+'
    message = f'{part5_prefix}{part5_total} GREEN AND {part6_prefix}{part6_total} ORANGE PARTS NEEDED TO START'
    if part5_count == part5_max and part6_count == part6_max:
        part_counts_ok = part_counts_ok + 1
    else:
        part_counts_ok = 0
    if part_counts_ok == min_validations:
        buzzer_verify()
        parts_counting = False
    window.instruction.config(text=message)
    return validations

def validate_step():
    global current_step
    global step_validations
    current = str(current_step)
    step_validations[current] = step_validations[current] + 1
    return current

def update_progress():
    global current_step
    global step_validations
    current = validate_step()
    if step_validations[current] == min_validations:
        buzzer_verify()
        window.bar_images[current_step].config(
            image=color_images[current_step], bg=colors_hex['yes'])
        window.bar_progress[current_step].config(
            image=progress_images['yes'], bg=colors_hex['yes'])
        current_step = current_step + 1
        window.bar_images[current_step].config(
            image=color_images[current_step])
        window.bar_progress[current_step].config(
            image=progress_images['aux'], bg=colors_hex['aux'])

def wait_message():
    global current_model
    global current_message
    global message_waitings
    current = str(current_model) + str(current_message)
    message_waitings[current] = message_waitings[current] + 1
    return current

def update_message():
    global message_waitings
    global current_message
    current = wait_message()
    if message_waitings[current] == waiting_frames:
        current_message = current_message + 1

def update_model():
    global message_waitings
    global current_message
    global current_model
    current = wait_message()
    if message_waitings[current] == waiting_frames:
        current_message = 0
        current_model = current_model + 1

def print_currently(count):
    if count > 0:
        text = 'Current detections in Step #%s:' % current_step
    else:
        text = 'No objects detected in Step #%s.' % current_step
    window.currently.config(text=text)

def print_detections(predictions):
    text = ''
    for prediction in predictions:
        text += prediction.__str__() + '\n'
    window.detections.config(text=text)

def print_inference_time(start, end):
    ms = (end-start)/1000
    if ms > 0:
        text = 'Infer: {:.1f}ms'.format(ms)
        window.inference.config(text=text)

def print_total_time(start):
    end = datetime.now().microsecond
    ms = (end-start)/1000
    if ms > 0:
        text = 'Total: {:.1f}ms'.format(ms)
        window.total.config(text=text)

def write_instruction(message):
    window.instruction.config(text=message)

def draw_detections(image, predictions):
    for prediction in predictions:
        if 'hole' in prediction.Label:
            image = draw_rectangle(image, prediction.Box, colors_bgr['aux'], 2)
        elif 'part' in prediction.Label:
            image = draw_rectangle(image, prediction.Box, colors_bgr['yes'], 4)
        elif 'error' in prediction.Label:
            image = draw_rectangle(image, prediction.Box, colors_bgr['no'], 6)
        elif 'Part4' in prediction.Label:
            image = draw_rectangle(image, prediction.Box, colors_bgr['aux'], 2)
    return image

def draw_part_count(image, validations):
    for validation in validations:
        image = draw_rectangle(image, validation.Detection.Box, validation.Color, validation.Thickness)
        x = int(round(validation.Detection.Box.Left * 480))
        y = int(round(validation.Detection.Box.Top * 480))
        cv2.putText(image, validation.Text, (x + 85, y + 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,  
            fontScale=0.9, thickness=2, color=validation.Color, bottomLeftOrigin=False)
    return image

def draw_validation(checking_value):
    window.validation.config(
        image=validation_images[checking_value], bg=colors_hex[checking_value])
    window.assembly.config(bg=colors_hex[checking_value])

def draw_rectangle(image, box, color, thick):
    x1 = int(round(box.Left * 480 + 80))
    y1 = int(round(box.Top * 480))
    x2 = x1 + int(round(box.Width * 480))
    y2 = y1 + int(round(box.Height * 480))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
    return image

def draw_completed(image):
    draw_validation('yes')
    completed = cv2.imread(completed_image)
    mask = cv2.imread(completed_mask, cv2.IMREAD_GRAYSCALE)
    background = cv2.bitwise_or(image, image, mask = mask)
    added_image = cv2.bitwise_or(background, completed)
    cv2_array = cv2.cvtColor(added_image, cv2.COLOR_BGR2RGBA)
    return cv2_array

def draw_gloves(image):
    draw_validation('no')
    gloves = cv2.imread(gloves_image)
    mask = cv2.imread(gloves_mask, cv2.IMREAD_GRAYSCALE)
    background = cv2.bitwise_or(image, image, mask = mask)
    added_image = cv2.bitwise_or(background, gloves)
    cv2_array = cv2.cvtColor(added_image, cv2.COLOR_BGR2RGBA)
    return cv2_array

def draw_caution(image):
    draw_validation('no')
    caution = cv2.imread(caution_image)
    mask = cv2.imread(caution_mask, cv2.IMREAD_GRAYSCALE)
    background = cv2.bitwise_or(image, image, mask = mask)
    added_image = cv2.bitwise_or(background, caution)
    cv2_array = cv2.cvtColor(added_image, cv2.COLOR_BGR2RGBA)
    return cv2_array

from threading import Thread

def leds_off():
    for led in leds:
        led.off()

def led_routing(text: str):
    if text == 'yes':
        led_green_on()
    elif text == 'aux':
        led_white_on()
    elif text == 'no':
        led_red_on()

def led_green_on():
    leds_off()
    led_green.on()

def led_white_on():
    leds_off()
    led_white.on()

def led_red_on():
    leds_off()
    led_red.on()

def buzzer_beep(time: float):
    buzzer.on()
    sleep(time)
    buzzer.off()

def buzzer_verify():
    buzzer.off()
    Thread(target=buzzer_beep, args=(0.200, ), daemon=True).start()

def main():
    video_streaming()
    window.root.mainloop()

if __name__ == '__main__':
    main()