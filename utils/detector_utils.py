# Utilities for object detector.
import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
detection_graph = tf.Graph()


# Trained Model working directory
TRAINED_MODEL_DIR = 'frozen_graphs'

# trained model path that is used for the object detection
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/faster_rcnn_frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/mask_detection_label_map.pbtxt'

# 'with_mask', 'without_mask', 'mask_weared_incorrect'
NUM_CLASSES = 3

# load label map using utils provided by tensorflow object detection API,
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)
print("category_index: \n", category_index)

# Load a frozen interference graph into memory
def load_inference_graph():
    # load frozen tensor-flow model into memory
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # od_graph_def = tf.GraphDef()
        od_graph_def = tf.compat.v1.GraphDef()

        # loading the .pb file to python Tensorflow model
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


a=b=0
id= ""
def draw_box_on_image(num_faces_detected, score_thresh,
                          scores, boxes, classes,
                          im_width, im_height, image_np):

    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    
    # To more easily differentiate distances and detected bboxes
    avg_width = 4.0

    global a,b, id
    faceDetectorCounter=0
    color = None
    
    color0 = (255,0,0) # red color
    color1 = (0,50,255) # blue color

    face_counter = []
    for i in range(num_faces_detected):
        if (scores[i] > score_thresh):
            if classes[i] == 1:
                id = 'with_mask'
                avg_width = 2.0

            if classes[i] == 2:
                id = 'without_mask'
                # to compensate bbox size change
                avg_width = 2.0

            if classes[i] == 3:
                id = 'mask_weared_incorrect'
                avg_width = 2.0

            
            if i == 0:
                color = color0
            else:
                color = color1

            """boxes represent normalized([x_min, y_min, x_max, y_max])
            which equates to:
            xmin/img_width, ymin/img_height, x_max/img_width, ymax/img_height
            """

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))


            # it will check if there is any image in-front of camera
            dist = distance_to_camera(avg_width, focalLength, int(right-left))

            if dist:
                print("dist--->>", dist)
                faceDetectorCounter += 1

            if faceDetectorCounter == 0:
                print("No Face detected...")
                b = 0

            else:
                print("Face detected...")
                b = 1

            cv2.rectangle(image_np, p1, p2, color , 3, 1)

            cv2.putText(image_np, 'face_detected '+ str(i)+ ':'+ id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
                        (int(im_width*0.65),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
           
            # a = alertcheck.drawboxtosafeline(image_np, p1, p2, Line_Position2, Orientation)

    return a, b, id

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

# compute and return the distance of face from the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')

    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    # add sample_size along with shape of the image
    image_np_expanded = np.expand_dims(image_np, axis=0) # image.shape 123, 123, 3 --->> 1, 123, 123, 3

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # print("boxes-->>", boxes[0][:10])
    # print("scores-->>", scores[0][:10])
    # print("classes->>", classes[0][:10])

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
