import argparse
import os
import time
import tkinter as tk
import winsound
from datetime import datetime
from multiprocessing import Process
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from object_detection.utils import label_map_util
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels

duration = 500  # millisecond
freq = 440  # Hz


# from pushbullet import Pushbullet
# pb1 = Pushbullet("o.vjA8zDopKpzNTsV9sBrZgmnXpJXrxHzY") #Pranav
# pb2 = Pushbullet("o.W3fAkB4Nv9ycPws2LIr2JeY2wuOPYgJl") #Mahesh


CWD_PATH = os.getcwd()

#VideoCam specifications
video_source = 0
width = 640
height = 480

#Data for GUI (Tkinter)
skipCount = 0
skipFlag = 0
image_count = 1
coco_classes = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl',
                'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork',
                'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter',
                'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign',
                'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
alert_classes = []
list_count = 0


def addToList():
    global list_count
    list_count += 1
    valFromMenu = drop_down_val.get()
    list_box.insert(list_count, valFromMenu)
    alert_classes.append(valFromMenu)
    list_box.select_set(0)


def removeFromList():
    global list_count
    if(list_count == 0):
        print("List is empty")
        return
    valFromList = list_box.curselection()
    #print(valFromList)
    list_box.delete(valFromList)
    del alert_classes[valFromList[0]]
    list_count -= 1
    list_box.select_set(0)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(
    CWD_PATH, 'models', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    CWD_PATH, 'object_detection', 'data', 'oid_bbox_trainable_label_map.pbtxt')

NUM_CLASSES = 545

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.7
    )

    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


def sendNotification(frame):
    winsound.Beep(freq, duration)
    currentTime = datetime.now().strftime("%d-%m-%y;%H-%M-%S")
    cv2.imwrite("Alert images/"+str(currentTime)+'.png', frame)
    #cv2.imwrite(str(currentTime)+'.png',frame)

    #push = pb.push_note("This is the title", "Warning")
    # with open("Alert images/"+str(currentTime)+".png", "rb") as pic:
    #     file_data = pb1.upload_file(pic, str(currentTime)+".jpeg")
    # pb1.push_file(**file_data)


def display():
    global skipCount, skipFlag
    frame = video_capture.read()
    input_q.put(frame)

    if output_q.empty():
        pass  # fill up queue
        display()
        fps.update()
        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        data = output_q.get()
        rec_points = data['rect_points']
        class_names = data['class_names']
        class_colors = data['class_colors']
        for point, name, color in zip(rec_points, class_names, class_colors):
            cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                          (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
            cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                          (int(point['xmin'] * width) + len(name[0]) * 6,
                           int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
            cv2.putText(frame, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), font,
                        0.3, (0, 0, 0), 1)
        if(skipFlag == 0):
            for name in zip(class_names):
                #print(str(name[0]).split("'",2)[1].split(':',1)[0].strip())
                for i in alert_classes:
                    if((str(name[0]).split("'", 2)[1].split(':', 1)[0].strip()) == i):
                        cv2.imshow('Alert', alert_img)
                        skipFlag = 1
                        #cv2.imwrite('Alert images/'+str(image_count)+'.png',frame)
                        t1 = Thread(target=sendNotification, args=(frame,))
                        t1.start()
                        """with open("Alert images/"+str(image_count)+".png", "rb") as pic:
                            file_data = pb.upload_file(pic, "Alert.jpeg")
                        pb.push_file(**file_data)"""
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        cam_output.imgtk = imgtk
        cam_output.configure(image=imgtk)
        skipCount += 1
        if(skipCount == 7):
            skipCount = 0
            skipFlag = 0
        cam_output.after(250, display)
        fps.update()
        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))


if __name__ == '__main__':
    video_source = 0
    width = 640
    height = 480
    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()

    root = tk.Tk()
    root.title("Object recognition")
    #root.geometry("765x480")

    # menu right
    menu_right = tk.Frame(root, width=200, height=480, bg="black")
    #menu_right_upper = tk.Frame(menu_right )#, height=240)
    #menu_right_lower = tk.Frame(menu_right)#, height=240)

    label1 = tk.Label(menu_right, text="Alert tags",
                      font=("TImes", 12, "bold"), bg="blue")
    label1.pack(fill=tk.X, pady=(0, 5))
    list_box = tk.Listbox(menu_right)
    list_box_button = tk.Button(
        master=menu_right, text="Remove", command=removeFromList)
    list_box.pack(fill=tk.X)
    list_box_button.pack(fill=tk.X, pady=(5, 0))
    label2 = tk.Label(menu_right, text="Select from",
                      font=("Times", 12, "bold"), bg="blue")
    label2.pack(fill=tk.X, pady=(20, 0))
    drop_down_val = tk.StringVar(menu_right)
    drop_down_val.set(coco_classes[0])  # default value
    drop_down_menu = tk.OptionMenu(menu_right, drop_down_val, *coco_classes)
    drop_down_button = tk.Button(
        master=menu_right, text="Add", command=addToList)
    drop_down_menu.pack(fill=tk.X, pady=(10, 0))
    drop_down_button.pack(fill=tk.X, pady=(5, 0))

    #menu_right_upper.grid(column=0,row=0)
    #menu_right_lower.grid(column=0,row=1)

    cam_output = tk.Label(root)

    cam_output.grid(column=0, row=0)
    menu_right.grid(row=0, column=1, sticky="nsew")

    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=video_source,
                                      width=width,
                                      height=height).start()

    fps = FPS().start()
    t = time.time()
    alert_img = cv2.imread('alert.jpg', cv2.IMREAD_COLOR)
    display()
    root.mainloop()
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    video_capture.stop()
    exit()
#activate tensorflow
#cd C:\Project\object_detector_app
#python object_detection_multithreading.py
