import cv2
import numpy as np

neural_net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg') #Create a Neural Network variable

classes = []   #Create a list to store the class names
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

img=cv2.imread('Test_Image (1).jpg')
height,width,_=img.shape

treat_img = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)  #swapRB changes BRG to RGB
neural_net.setInput(treat_img)

output_layers_names=neural_net.getUnconnectedOutLayersNames()
layerOutputs=neural_net.forward(output_layers_names)

bounding_boxes=[]
probabilities=[]
class_labels=[]

for output in layerOutputs:
    for detection in output:
        prob_values = detection[5:]#First Four=Location of Box #Fifth=Accuracy of Bounding Box #Rest 80=Probabilities corresponding to each class
        class_label = np.argmax(prob_values)
        class_probabilitiy = prob_values[class_label]
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

indexes = cv2.dnn.NMSBoxes(bounding_boxes,probabilities,0.2,0.4) #Non Maximum Suppressions

#Using NMS will remove all redundant bounding boxes

font = cv2.FONT_HERSHEY_PLAIN

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



cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
