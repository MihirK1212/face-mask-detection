import cv2
import numpy as np

neural_net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap=cv2.VideoCapture(1)

if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    treat_img= cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    neural_net.setInput(treat_img)
    output_layers_names = neural_net.getUnconnectedOutLayersNames()
    layerOutputs = neural_net.forward(output_layers_names)

    bounding_boxes = []
    probabilities = []
    class_labels = []

    for output in layerOutputs:
        for detection in output:
            prob_values=detection[5:]
            class_label=np.argmax(prob_values)
            class_probabilitiy=prob_values[class_label]
            if class_probabilitiy > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bounding_boxes.append([x, y, w, h])
                probabilities.append((float(class_probabilitiy)))
                class_labels.append(class_label)

    indexes = cv2.dnn.NMSBoxes(bounding_boxes, probabilities, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = bounding_boxes[i]
            label = str(classes[class_labels[i]])
            class_probabilitiy =str((round(probabilities[i],2))*100)
            if class_labels[i]==0:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, label, (x, y+20), font, 1, (0,255,0), 1)
            else:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(img, label, (x, y+20), font, 1, (0,0,255), 1)


    cv2.imshow('Webcam For Face Mask Detection', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
