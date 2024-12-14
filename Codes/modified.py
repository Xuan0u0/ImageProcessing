import cv2

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
    new_image = image.copy()

    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        label = '{}: {:.2f}'.format(names[int(classid)], float(conf))
        cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 225, 0), 2)
        cv2.putText(new_image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 225, 0), 2, cv2.LINE_AA)

    return new_image

if __name__ == "__main__":
    model, names = initNet()

    image_path = 'test2.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print("無法讀取圖片，請確認路徑正確或檔案是否存在。")
        exit()

    image = cv2.resize(image, (640, 640))
    classes, confs, boxes = nnProcess(image, model)
    output_image = drawBox(image, classes, confs, boxes, names)

    cv2.imwrite('output2.jpg', output_image)
    print("影像處理完成。")
