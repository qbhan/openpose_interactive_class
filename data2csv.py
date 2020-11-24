# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import json
import numpy as np
import csv
import ast
import threading

root_dir = "D:\projects\openpose\examples\media/Gesture"
gestures = next(os.walk(root_dir))[1]
Hand = {}

lock = threading.Lock()

def resize_image(image_path, scale):
    imageToProcess = cv2.imread(image_path)
    h, w, c = imageToProcess.shape
    new_size = int(max(h, w))
    color = (0, 0, 0)
    background = np.full((new_size, new_size, c), color, dtype=np.uint8)
    ww = (new_size - w) // 2
    hh = (new_size - h) // 2
    background[hh:hh + h, ww:ww + w] = imageToProcess
    imageToProcess = background
    h, w = imageToProcess.shape[0], imageToProcess.shape[1]
    imageToProcess = cv2.resize(imageToProcess, (scale * w, scale * h), interpolation=cv2.INTER_LANCZOS4)
    h, w = imageToProcess.shape[0], imageToProcess.shape[1]
    return imageToProcess, (h, w)

class gestureThread (threading.Thread):
    def __init__(self, threadID, gesture, folders):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.gesture = gesture
        self.folders = folders

    def run(self):
        global lock
        gesture = self.gesture

        field_names = ['id', 'left', 'right']
        with open('results/result_' + str(gesture) + '_'+ str(self.threadID) + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            folders = self.folders
            for folder_id, folder in enumerate(folders):
                # x[1] denotes the files while scanning through the folder.
                images = [x[2] for x in os.walk(root_dir + "/" + gesture + "/" + folder)][0]
                for image in images:
                    # print(os.path.join(root_dir, gesture, folder, image))
                    if image == '.DS_Store':
                        continue

                        # Read image and face rectangle locations
                    image_path = os.path.join(root_dir, gesture, folder, image)
                    imageToProcess, (h, w) = resize_image(image_path, 5)
                    # print(h, w)
                    box = min(h, w)
                    handRectangles = [
                        # Left/Right hands person 0
                        [
                            op.Rectangle(100., 100., box, box),
                            op.Rectangle(0., 0., box, box),
                        ],
                    ]

                    # Create new datum
                    datum = op.Datum()
                    datum.cvInputData = imageToProcess
                    datum.handRectangles = handRectangles

                    # Process and display image
                    opWrapper.emplaceAndPop([datum])
                    hand = {}
                    hand['id'] = folder + '_' + image[:-4]
                    left = str(datum.handKeypoints[0]).replace('\n', ',')
                    right = str(datum.handKeypoints[1]).replace('\n', ',')
                    # print("Left hand keypoints: \n" + left)
                    hand['left'] = left
                    # print("Right hand keypoints: \n" + right)
                    hand['right'] = right
                    # print('and Keypoints : \n', hand)
                    # lock.acquire()
                    writer.writerow(hand)
                    # lock.release()
                    # print(hand)

                    print(os.path.join(root_dir, gesture, folder, image), "written.")
            print('FINISH THREAD {}'.format(self.gesture))


class gestureAllThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.gesture = name
        self.counter = counter
    def run(self):
        print("got in", self.gesture, '\n')
        gesture = self.gesture
        folders = next(os.walk(root_dir + "/" + gesture))[1]

        field_names = ['id', 'left', 'right']
        with open('results/result_' + 'new ' + str(gesture) + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            for folder_id, folder in enumerate(folders):
                # x[1] denotes the files while scanning through the folder.
                images = [x[2] for x in os.walk(root_dir + "/" + gesture + "/" + folder)][0]

                for image in images:
                    # print(os.path.join(root_dir, gesture, folder, image))
                    if image == '.DS_Store':
                        continue

                        # Read image and face rectangle locations
                    image_path = os.path.join(root_dir, gesture, folder, image)
                    imageToProcess, (h, w) = resize_image(image_path, 5)
                    print(h, w)
                    box = min(h, w)
                    handRectangles = [
                        # Left/Right hands person 0
                        [
                            op.Rectangle(100., 100., box, box),
                            op.Rectangle(0., 0., box, box),
                        ],
                    ]

                    # Create new datum
                    datum = op.Datum()
                    datum.cvInputData = imageToProcess
                    datum.handRectangles = handRectangles

                    # Process and display image
                    opWrapper.emplaceAndPop([datum])
                    hand = {}
                    hand['id'] = folder + '_' + image[:-4]
                    left = str(datum.handKeypoints[0]).replace('\n', ',')
                    right = str(datum.handKeypoints[1]).replace('\n', ',')
                    # print("Left hand keypoints: \n" + left)
                    hand['left'] = left
                    # print("Right hand keypoints: \n" + right)
                    hand['right'] = right
                    # print('and Keypoints : \n', hand)
                    writer.writerow(hand)
                    print(hand)

                    print(os.path.join(root_dir, gesture, folder, image), "working in progress.")
        print('FINISH THREAD {}'.format(self.threadID))

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 0
    params["disable_blending"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # START THREADS
    # thread1 = gestureAllThread(0, gestures[0], 0)
    # thread2 = gestureAllThread(1, gestures[1], 1)
    # thread3 = gestureAllThread(2, gestures[2], 2)
    # thread4 = gestureAllThread(3, gestures[3], 3)
    # thread5 = gestureAllThread(4, gestures[4], 4)
    #
    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()
    # thread5.start()

    # threads for gesture1
    folders = next(os.walk(root_dir + "/" + gestures[0]))[1]
    length = len(folders)
    print(length)
    thread_num = 5
    sublen = length//thread_num
    rem = length % thread_num
    print(sublen)
    threads = []
    start = 0
    for i in range(thread_num):
        threads.append(gestureThread(i, gestures[0], folders[start:start + sublen if i != thread_num-1 else length+rem]))
        print(len(folders[start:start + sublen if i != thread_num - 1 else length + rem]))
        start += sublen


    for i in range(thread_num):
        threads[i].start()

    print('Exiting Main Thread')

except Exception as e:
    print(e)
    sys.exit(-1)