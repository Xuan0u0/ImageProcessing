import cv2
import time

turnX = 'stop'
turnY = 'stop'
detected = False
WIDTH = 0
HEIGHT = 0
frame_resize = None

def initNet():
    CONFIG = 'yolov4/yolov4-tiny.cfg'
    WEIGHT = 'yolov4/yolov4-tiny.weights'
    NAMES = 'yolov4/obj.names'

    with open(NAMES, 'r') as f:
        names = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(CONFIG, WEIGHT)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0)
    model.setInputSwapRB(True)

    return model, names

def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.6, 0.3)
    return classes, confs, boxes

def drawBox(image, classes, confs, boxes, names):
    global turnX, turnY, WIDTH, HEIGHT, detected
    new_image = image.copy()

    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        label = '{}: {:.2f}'.format(names[int(classid)], float(conf))

        if classid == 0:
            detected = True
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 225, 0), 2)
            cv2.putText(new_image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 225, 0), 2, cv2.LINE_AA)
            if (x + w // 2) > (WIDTH // 2) + 70:
                turnX = 'left'
            elif (x + w // 2) < (WIDTH // 2) - 70:
                turnX = 'right'
            else:
                turnX = 'stop'

            if (y + h // 2) > (HEIGHT // 2) + 50:
                turnY = 'up'
            elif (y + h // 2) < (HEIGHT // 2) - 50:
                turnY = 'down'
            else:
                turnY = 'stop'
        else:
            detected = False

    return new_image

def get_frame():
    global WIDTH, HEIGHT, frame_resize
    cap = cv2.VideoCapture(0)
    ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    WIDTH = 480
    HEIGHT = int(WIDTH / ratio)

    while True:
        ret, frame = cap.read()
        frame_resize = cv2.resize(frame, (WIDTH, HEIGHT))
        if cv2.waitKey(1) == 27:
            break

def detect():
    global frame_resize
    model, names = initNet()

    while True:
        begiin_time = time.time()
        if frame_resize is not None:
            classes, confs, boxes = nnProcess(frame_resize, model)
            frame2 = drawBox(frame_resize, classes, confs, boxes, names)
            fps = 'fps: {:.2f}'.format(1 / (time.time() - begiin_time))
            cv2.putText(frame2, fps, (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 204, 255), 2)
            cv2.imshow('video', frame2)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
