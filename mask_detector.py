import cv2
import argparse
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy 
import numpy as np

lst1=[]
lst2=[]
individual_class_counter = []

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')

args = vars(ap.parse_args())

# loading the .pb model into tensorflow session & graph
detection_graph, sess = detector_utils.load_inference_graph()

def save_data(face_detected_with_mask, face_detected_without_mask):

    try:
        ## PULLING OUT PREVIOUS RECORDED DATA

        # xlrd book object
        rb = xlrd.open_workbook('result.xls')

        # xlrd sheet object, grabbing the 1st sheet by index
        sheet = rb.sheet_by_index(0)
        sheet.cell_value(0, 0)

        # pull-out the last recorded date from excel, sheet.nrows gives total filled rows
        # indexing starts from 0, then last_row = sheet.nrows -1
        last_recorded_date = sheet.cell_value(sheet.nrows-1, 1)
        print("last_recorded_date-->>", last_recorded_date)

        ## TO UPDATE NEW DATA TO THE EXCEL FILE, EARLIER WE HAVE PULLED-OUT PREVIOUS DATA
        rb = xlrd.open_workbook('result.xls')

        wb = copy(rb)
        w_sheet = wb.get_sheet(0)

        # to add current date to the results.xls
        today = date.today()
        today = str(today)

        # checking last_recorded date and today's date
        if last_recorded_date==today:
            # pulling out last data filled under same date
            data_col2 = sheet.cell_value(sheet.nrows-1, 2)
            data_col3 = sheet.cell_value(sheet.nrows-1, 3)

            # updating the previous data of the same day..
            print("face_detected_with_mask---->>", face_detected_with_mask)
            print("face_detected_without_mask-->>",face_detected_without_mask)

            w_sheet.write(sheet.nrows-1, 2, data_col2 + face_detected_with_mask)
            w_sheet.write(sheet.nrows-1, 3, data_col3 + face_detected_without_mask)
            wb.save('result.xls')

        else:
            ## new date recorded..
            w_sheet.write(sheet.nrows, 0, sheet.nrows)
            w_sheet.write(sheet.nrows, 1, today)
            w_sheet.write(sheet.nrows, 2, face_detected_with_mask)
            w_sheet.write(sheet.nrows, 3, face_detected_without_mask)
            wb.save('result.xls')

    except FileNotFoundError:
        today = date.today()
        today=str(today)

        # Workbook is created
        wb = Workbook()

        # add_sheet is used to create sheet.
        sheet = wb.add_sheet('Sheet 1')

        sheet.write(0, 0, 'SR#')
        sheet.write(0, 1, 'Date')
        sheet.write(0, 2, 'face detected with mask')
        sheet.write(0, 3, 'face detected without mask')

        idx = 1
        sheet.write(1, 0, idx)
        sheet.write(1, 1, today)
        sheet.write(1, 2, face_detected_with_mask)
        sheet.write(1, 3, face_detected_without_mask)

        wb.save('result.xls')
        
if __name__ == '__main__':

    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80
    
    #vs = cv2.VideoCapture('rtsp://192.168.1.64')
    vs = VideoStream(0).start()

    # Orientation of machine << lr,rl,bt,tb>>
    # input("Enter the orientation of machine----->>: ")
    Orientation= 'bt'

    # function to count the occurrence of face detected
    def count_no_of_times(lst):
        x = y = cnt = 0
        for i in lst:
            x = y
            y = i
            if x == 0 and y == 1:
                cnt = cnt + 1
        return cnt

    # max number of faces we want to detect/track in one screen
    max_faces_to_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0
    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if im_height is None:
                im_height, im_width = frame.shape[:2]

            try:
                # Convert image to rgb since openCV loads images in bgr, if not accuracy will decrease
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensor-flow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)


            # Draw bounding boxes and text
            a, b, id = detector_utils.draw_box_on_image(
                max_faces_to_detect, score_thresh, scores, boxes, classes,
                im_width, im_height, frame,
            )

            individual_class_counter.append(id)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

            fps = num_frames/elapsed_time

            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    vs.stop()
                    break

        print("Classes recorded after Obj Detection--->>", individual_class_counter)
        individual_class_counter = [ class_ for class_ in individual_class_counter if class_ != '']
        print("After removal of no class detection results--\n", individual_class_counter)

        ## to count the occurence of individual classes_
        d = {}
        classRecorded = []
        for idx, face_label in enumerate(individual_class_counter):
            if idx == 0:
                d[face_label] = idx
                classRecorded.append(face_label)

            else:
                if face_label not in d:
                    classRecorded.append(face_label)
                    d.clear()
                    d[face_label] = idx

        print(classRecorded)

        face_with_mask_detected = classRecorded.count("with_mask")
        face_without_mask_detected = classRecorded.count("without_mask")
        face_with_incorrect_mask_detected = classRecorded.count("mask_weared_incorrect")

        print("face_with_mask_detected", face_with_mask_detected)
        print("face_without_mask_detected", face_without_mask_detected)
        print("face_with_incorrect_mask_detected", face_with_incorrect_mask_detected)

        save_data(face_with_mask_detected, face_without_mask_detected)
        print("Average FPS: ", str("{0:.2f}".format(fps)))
        
    except KeyboardInterrupt:

        face_with_mask_detected = classRecorded.count("with_mask")
        face_without_mask_detected = classRecorded.count("without_mask")
        today = date.today()
        save_data(face_with_mask_detected, face_without_mask_detected)
        # save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))