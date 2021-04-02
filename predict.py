import os,time,cv2, sys, math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

from yolo.YOLO import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--gt_image', type=str, default='', help='The GT image you want to evaluate on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid/", required=False, help='The dataset you are using')
parser.add_argument('--yolo', type=bool, default=True, help='If YOLO is used in generating the final mask')
parser.add_argument('--yolo_weights', type=str, default='/content/Sem_Seg_Suite/yolo/yolov3.weights', help='YOLO pretrained weights directory')
parser.add_argument('--yolo_cfg', type=str, default='/content/Sem_Seg_Suite/yolo/yolov3.cfg', help='YOLO pretrained cfg file directory')
parser.add_argument('--coco_names', type=str, default='/content/Sem_Seg_Suite/yolo/coco.names', help='COCO dataset classes file directory')
parser.add_argument('--Alpha', type=float32, default=0.5, help='IoU threshold to create Final Mask')
args = parser.parse_args()

def visualize_results(accuracy, class_accuracies, prec, rec, f1, iou, run_time):
    print("Prediction accuracy = ", accuracy)
    print("Per class prediction accuracies =")
    print(f"Object\t= {class_accuracies[0]}")
    print(f"Map\t= {class_accuracies[1]}")
    print("Precision = ", prec)
    print("Recall = ", rec)
    print("F1 score = ", f1)
    print("IoU score = ", iou)
    print("Run time = ", run_time)

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


print("Testing image " + args.image)

loaded_image = utils.load_image(args.image)
resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

st = time.time()
output_image = sess.run(network,feed_dict={net_input:input_image})

run_time = time.time()-st

output_image = np.array(output_image[0,:,:,:])
output_image = helpers.reverse_one_hot(output_image)

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(args.image)
cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
print("Finished!")
print("Wrote image " + "%s_pred.png"%(file_name))

if args.gt_image:
    gt_image = cv2.resize(utils.load_image(args.gt_image), (args.crop_width, args.crop_height))
    gt_image = helpers.reverse_one_hot(helpers.one_hot_it(gt_image, label_values))
    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt_image, num_classes=num_classes)
    visualize_results(accuracy, class_accuracies, prec, rec, f1, iou, run_time)

if args.yolo:
    full_mask_st = time.time()

    YOLO_obj = YOLO(args.yolo_weights, args.yolo_cfg, args.coco_names, resized_image)
    YOLO_obj.Detect()
    yolo_boxes = YOLO_obj.boxes
    full_mask_image = utils.create_final_mask(output_image, yolo_boxes, args.Alpha)

    full_mask_run_time = time.time()-full_mask_st

    full_mask_vis_image = helpers.colour_code_segmentation(full_mask_image, label_values)
    cv2.imwrite("%s_full_mask.png"%(file_name),cv2.cvtColor(np.uint8(full_mask_vis_image), cv2.COLOR_RGB2BGR))

    print("")
    print("Full Mask Finished!")
    print("Wrote full mask image " + "%s_full_mask.png"%(file_name))

    if args.gt_image:
        full_mask_accuracy, full_mask_class_accuracies, full_mask_prec, full_mask_rec, full_mask_f1, full_mask_iou = \
        utils.evaluate_segmentation(pred=full_mask_image, label=gt_image, num_classes=num_classes)
        visualize_results(full_mask_accuracy, full_mask_class_accuracies, full_mask_prec, full_mask_rec, \
                        full_mask_f1, full_mask_iou, full_mask_run_time)

        acc_diff = full_mask_accuracy - accuracy
        acc_imp = (acc_diff/accuracy) * 100
        map_acc_diff = full_mask_class_accuracies[1] - class_accuracies[1]
        map_acc_imp = (map_acc_diff/class_accuracies[1]) * 100
        obj_acc_diff = full_mask_class_accuracies[0] - class_accuracies[0]
        obj_acc_imp = (obj_acc_diff/class_accuracies[0]) * 100
        prec_diff = full_mask_prec - prec
        prec_imp = (prec_diff/prec) * 100
        rec_diff = full_mask_rec - rec
        rec_imp = (rec_diff/rec) * 100
        f1_diff = full_mask_f1 - f1
        f1_imp = (f1_diff/f1) * 100
        iou_diff = full_mask_iou - iou
        iou_imp = (iou_diff/iou) * 100

        print("\n==== Comparison ====\t(difference),\t\t\t(improvement perc %)")
        print(f"Prediction accuracy\t=> ({acc_diff}),\t({acc_imp} %)")
        print("Per class prediction accuracies:")
        print(f"Object\t\t\t=> ({obj_acc_diff}),\t({obj_acc_imp} %)")
        print(f"Map\t\t\t=> ({map_acc_diff}),\t({map_acc_imp} %)")
        print(f"Precision\t\t=> ({prec_diff}),\t({prec_imp} %)")
        print(f"Recall\t\t\t=> ({rec_diff}),\t({rec_imp} %)")
        print(f"F1 score\t\t=> ({f1_diff}),\t({f1_imp} %)")
        print(f"IoU score\t\t=> ({iou_diff}),\t({iou_imp} %)")
