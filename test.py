import os,time,cv2, sys, math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

from yolo.YOLO import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid/", required=False, help='The dataset you are using')
parser.add_argument('--main_dir', type=str, default="/content/drive/MyDrive/Thesis/ModelTraining/", help='The main dir where the training outcome will be saved')
parser.add_argument('--yolo', type=bool, default=True, help='If YOLO is used in generating the final mask')
parser.add_argument('--yolo_weights', type=str, default='/content/Sem_Seg_Suite/yolo/yolov3.weights', help='YOLO pretrained weights directory')
parser.add_argument('--yolo_cfg', type=str, default='/content/Sem_Seg_Suite/yolo/yolov3.cfg', help='YOLO pretrained cfg file directory')
parser.add_argument('--coco_names', type=str, default='/content/Sem_Seg_Suite/yolo/coco.names', help='COCO dataset classes file directory')
parser.add_argument('--custom_dir', type=str, default="", help='Add custom directory (e.g Google Colab)')
args = parser.parse_args()

#Define useful functions
def write_to_target(target, file_name, class_accuracies, accuracy, prec, rec, f1, iou):
    target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
    for item in class_accuracies:
        target.write(", %f"%(item))
    target.write("\n")

def calculate_avg(scores_list, class_scores_list, precision_list, recall_list, f1_list, iou_list, run_times_list, score_title = 'Seg Mask Results'):
    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    avg_time = np.mean(run_times_list)
    print('====================')
    print(score_title)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)
    print("Average run time = ", avg_time)

    return avg_score, class_avg_scores, avg_precision, avg_recall, avg_f1, avg_iou

# Get the names of the classes so we can record the evaluation results
DS_dir = args.custom_dir + args.dataset
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(DS_dir, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

main_dir = args.main_dir

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=DS_dir)

# Create directories if needed
if not os.path.isdir("%s%s"%(main_dir, "Test")):
        os.makedirs("%s%s"%(main_dir, "Test"))

target=open("%s/test_scores.csv"%("/content/drive/MyDrive/Thesis/ModelTraining/Test"),'w')
target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []

if args.yolo:
    full_mask_target=open("%s/test_scores_full_mask.csv"%("/content/drive/MyDrive/Thesis/ModelTraining/Test"),'w')
    full_mask_target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
    full_mask_scores_list = []
    full_mask_class_scores_list = []
    full_mask_precision_list = []
    full_mask_recall_list = []
    full_mask_f1_list = []
    full_mask_iou_list = []
    full_mask_run_times_list = []

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    image = np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width])
    input_image = np.expand_dims(image, axis=0)/255.0
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_time = time.time()-st
    run_times_list.append(run_time)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

    file_name = utils.filepath_to_name(test_input_names[ind])
    write_to_target(target, file_name, class_accuracies, accuracy, prec, rec, f1, iou)
    # target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
    # for item in class_accuracies:
    #     target.write(", %f"%(item))
    # target.write("\n")

    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)

    gt_vis = helpers.colour_code_segmentation(gt, label_values)

    cv2.imwrite("%s/%s_pred.png"%("Test", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt_vis), cv2.COLOR_RGB2BGR))

    if args.yolo:
        full_mask_st = time.time()
        YOLO_obj = YOLO(args.yolo_weights, args.yolo_cfg, args.coco_names, image)
        YOLO_obj.Detect()
        yolo_boxes = YOLO_obj.boxes
        full_mask_image = utils.create_final_mask(output_image, yolo_boxes)

        full_mask_run_time = time.time()-full_mask_st
        full_mask_run_times_list.append(full_mask_run_time)

        # One layer is enough for evaluation (Binary mask)
        full_mask_accuracy, full_mask_class_accuracies, full_mask_prec, full_mask_rec, full_mask_f1, full_mask_iou = \
        utils.evaluate_segmentation(pred=full_mask_image, label=gt, num_classes=num_classes)

        write_to_target(full_mask_target, file_name, full_mask_class_accuracies, full_mask_accuracy, full_mask_prec, \
                        full_mask_rec, full_mask_f1, full_mask_iou)

        full_mask_scores_list.append(full_mask_accuracy)
        full_mask_class_scores_list.append(full_mask_class_accuracies)
        full_mask_precision_list.append(full_mask_prec)
        full_mask_recall_list.append(full_mask_rec)
        full_mask_f1_list.append(full_mask_f1)
        full_mask_iou_list.append(full_mask_iou)

        cv2.imwrite("%s/%s_full_mask.png"%("Test", file_name), full_mask_image)
        # full_mask_target.write("%s, %f, %f, %f, %f, %f"%(file_name, full_mask_accuracy, full_mask_prec, \
        #                                                 full_mask_rec, full_mask_f1, full_mask_iou))
        # for item in full_mask_class_accuracies:
        #     full_mask_target.write(", %f"%(item))
        # full_mask_target.write("\n")


target.close()
avg_score, class_avg_scores, avg_precision, avg_recall, avg_f1, avg_iou = \
calculate_avg(scores_list, class_scores_list, precision_list, recall_list, f1_list, iou_list, run_times_list)

if args.yolo:
    full_mask_avg_score, full_mask_class_avg_scores, full_mask_avg_precision, full_mask_avg_recall, full_mask_avg_f1, full_mask_avg_iou = \
    calculate_avg(full_mask_scores_list, full_mask_class_scores_list, full_mask_precision_list, full_mask_recall_list, \
                full_mask_f1_list, full_mask_iou_list, full_mask_run_times_list, 'Full Mask Results')
    full_mask_target.close()

    acc_diff = full_mask_avg_score - avg_score
    acc_imp = (acc_diff/avg_score) * 100
    map_acc_diff = full_mask_class_avg_scores[1] - class_avg_scores[1]
    map_acc_imp = (map_acc_diff/class_avg_scores[1]) * 100
    obj_acc_diff = full_mask_class_avg_scores[0] - class_avg_scores[0]
    obj_acc_imp = (obj_acc_diff/class_avg_scores[0]) * 100
    prec_diff = full_mask_avg_precision - avg_precision
    prec_imp = (prec_diff/avg_precision) * 100
    rec_diff = full_mask_avg_recall - avg_recall
    rec_imp = (rec_diff/avg_recall) * 100
    f1_diff = full_mask_avg_f1 - avg_f1
    f1_imp = (f1_diff/avg_f1) * 100
    iou_diff = full_mask_avg_iou - avg_iou
    iou_imp = (iou_diff/avg_iou) * 100

    print("\n==== Comparison ====\t(difference),\t\t\t(improvement perc %)")
    print(f"Prediction accuracy\t=> ({acc_diff}),\t({acc_imp} %)")
    print("Per class prediction accuracies:")
    print(f"Object\t\t\t=> ({obj_acc_diff}),\t({obj_acc_imp} %)")
    print(f"Map\t\t\t=> ({map_acc_diff}),\t({map_acc_imp} %)")
    print(f"Precision\t\t=> ({prec_diff}),\t({prec_imp} %)")
    print(f"Recall\t\t\t=> ({rec_diff}),\t({rec_imp} %)")
    print(f"F1 score\t\t=> ({f1_diff}),\t({f1_imp} %)")
    print(f"IoU score\t\t=> ({iou_diff}),\t({iou_imp} %)")

# avg_score = np.mean(scores_list)
# class_avg_scores = np.mean(class_scores_list, axis=0)
# avg_precision = np.mean(precision_list)
# avg_recall = np.mean(recall_list)
# avg_f1 = np.mean(f1_list)
# avg_iou = np.mean(iou_list)
# avg_time = np.mean(run_times_list)
# print("Average test accuracy = ", avg_score)
# print("Average per class test accuracies = \n")
# for index, item in enumerate(class_avg_scores):
#     print("%s = %f" % (class_names_list[index], item))
# print("Average precision = ", avg_precision)
# print("Average recall = ", avg_recall)
# print("Average F1 score = ", avg_f1)
# print("Average mean IoU score = ", avg_iou)
# print("Average run time = ", avg_time)
