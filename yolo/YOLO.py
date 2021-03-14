###https://www.codespeedy.com/yolo-object-detection-from-image-with-opencv-and-python/

import cv2
import numpy as np
from PIL import Image

#All COCO Classes
# ['person' [0]*, 'bicycle' [1]*, 'car' [2]*, 'motorbike' [3]*, 'aeroplane' [4]*, 'bus' [5]*, 'train' [6]*, 'truck' [7]*, 'boat' [8]*, 'traffic light' [9],
# 'fire hydrant' [10], 'stop sign' [11], 'parking meter' [12], 'bench' [13], 'bird' [14]*, 'cat' [15]*, 'dog' [16]*, 'horse' [17]*, 'sheep' [18]*, 'cow' [19]*,
# 'elephant' [20]*, 'bear' [21]*, 'zebra' [22]*, 'giraffe' [23]*, 'backpack' [24]*, 'umbrella' [25]*, 'handbag' [26]*, 'tie' [27], 'suitcase' [28]*, 'frisbee' [29]*,
# 'skis' [30], 'snowboard' [31], 'sports ball' [32]*, 'kite' [33]*, 'baseball bat' [34], 'baseball glove' [35], 'skateboard' [36], 'surfboard' [37], 'tennis racket' [38],
# 'bottle' [39], 'wine glass' [40], 'cup' [41], 'fork' [42], 'knife' [43], 'spoon' [44], 'bowl' [45], 'banana' [46], 'apple' [47], 'sandwich' [48], 'orange' [49],
# 'broccoli' [50], 'carrot' [51], 'hot dog' [52], 'pizza' [53], 'donut' [54], 'cake' [55], 'chair' [56]*, 'sofa' [57]*, 'pottedplant' [58]*, 'bed' [59],
# 'diningtable' [60]*, 'toilet' [61], 'tvmonitor' [62], 'laptop' [63], 'mouse' [64], 'remote' [65], 'keyboard' [66], 'cell phone' [67], 'microwave' [68], 'oven' [69],
# 'toaster' [70], 'sink' [71], 'refrigerator' [72], 'book' [73], 'clock' [74], 'vase' [75]*, 'scissors' [76], 'teddy bear' [77], 'hair drier' [78], 'toothbrush' [79]]


OBJECTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 32, 33, 56, 57, 58, 60, 75}

class YOLO:
    def __init__(self, Weights, Cfg, Coco_names, Input, Output = None):
        self.Weights = Weights
        self.Cfg = Cfg
        self.Coco_names = Coco_names
        self.Input = Input
        self.Output = Output
        self.img = None
        self.boxes = []

    def Detect(self):
        net = self.LoadYolo()
        classes = self.ParseClasses()
        output_layers = self.DefineLayers(net)
        input_type = type(self.Input)
        if input_type is str:
            self.ReadImage()
        elif input_type is np.ndarray:
            self.img = self.Input
        outs = self.ExtratFeat(net, output_layers)
        boxes, indexes, class_ids = self.ObjDetection(outs)
        self.DrawBoxes(boxes, indexes)


    def LoadYolo(self):
        #Load YOLO Algorithm
        net=cv2.dnn.readNet(self.Weights, self.Cfg)
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
                confidence = scores[class_id]         #3

                if class_id not in OBJECTS: continue

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
        return boxes, indexes, class_ids

    def DrawBoxes(self, boxes, indexes, with_label = None, classes = None, class_ids = None):
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
