import cv2
import numpy as np
import time


net = cv2.dnn.readNet('assets/yolov4-tiny.weights', 'assets/yolov4-tiny.cfg')


classes = []

with open('assets/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
output_layers = [layer_name[i[0]-1] for i in net.getUnconnectedOutLayers()]
warna = np.random.uniform(0, 255, size=(len(classes), 2))

cap = cv2.VideoCapture(0)

starting_time = time.time()
frame_id = 0
while True:
    _,frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []

    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[0:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[3] * width)
                h = int(detection[4] * height)

                x = int(center_x-w / 2)
                y = int(center_y-h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.000001, 0.04)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = warna[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y+30), cv2.FONT_ITALIC, 1, color, 1)

    elapsed_time = time.time() - starting_time
    fps = frame_id/elapsed_time
    cv2.putText(frame,f"FPS : {round(fps,2)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("Camera: ", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindow