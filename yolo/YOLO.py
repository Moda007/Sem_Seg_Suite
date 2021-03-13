###https://www.codespeedy.com/yolo-object-detection-from-image-with-opencv-and-python/

import cv2
import numpy as np
from PIL import Image

class YOLO:
    def __init__(self, Weights, Cfg, Coco_names, Input, Output):
        self.Weights = Weights
        self.Cfg = Cfg
        self.Coco_names = Coco_names
        self.Input = Input
        self.Output = Output
        self.img = None
        self.boxes = []

    def Detect(self):
        net = LoadYolo()
        classes = ParseClasses()
        output_layers = DefineLayers(net)
        ReadImage()
        outs = ExtratFeat(net, output_layers)
        boxes, indexes = ObjDetection(outs)
        DrawBoxes(boxes, indexes)


    def LoadYolo(self):
        #Load YOLO Algorithm
        net=cv2.dnn.readNet(self.Cfg)
        return net

    def ParseClasses(self):
        #To load all objects that have to be detected
        classes=[]
        with open(self.Coco_names,"r") as f:
            read=f.readlines()
        for i in range(len(read)):
            classes.append(read[i].strip("\n"))
        return classes

    def DefineLayers(self, net):
        #Defining layer names
        layer_names=net.getLayerNames()
        output_layers=[]
        for i in net.getUnconnectedOutLayers():
            output_layers.append(layer_names[i[0]-1])
        return output_layers

    def ReadImage(self):
        #Loading the Image
        self.img = cv2.imread(self.Input)
        # height,width,channels=img.shape
        # return height, width, channels

    def ExtratFeat(self, net, output_layers):
        #Extracting features to detect objects (magic parameters)
        blob = cv2.dnn.blobFromImage(self.img, 0.00392, (416,416), (0,0,0), True, crop=False)
                                                                #Inverting blue with red
                                                                #bgr->rgb
        #We need to pass the img_blob to the algorithm
        net.setInput(blob)
        outs = net.forward(output_layers)
        #print(outs)
        return outs

    def ObjDetection(self, outs):
        #Displaying informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        height, width, _ = self.img.shape
        for output in outs:
            for detection in output:
                #Detecting confidence in 3 steps
                scores = detection[5:]                #1
                class_id = np.argmax(scores)          #2
                confidence = scores[class_id]        #3

                if confidence > 0.5: #Means if the object is detected
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    #Drawing a rectangle
                    x = int(center_x-w/2) # top left value
                    y = int(center_y-h/2) # top left value

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                   #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        #Removing Double Boxes
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)
        return boxes, indexes

    def DrawBoxes(self, boxes, indexes, with_label = None, classes = None):
        self.out_img = self.img.copy()
        for i in range(len(boxes)):
            if i in indexes:
                self.boxes.append(boxes[i])
                x, y, w, h = boxes[i]
                if with_label: label = classes[class_ids[i]]  # name of the objects
                cv2.rectangle(self.out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if with_label: cv2.putText(self.out_img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        if self.Output:
            self.out_img = Image.fromarray(self.out_img)
            self.out_img.save(self.Output)
