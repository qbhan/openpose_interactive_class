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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class gestureThread (threading.Thread):
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
        with open('results/result_' + str(gesture) + '.csv', 'w', newline='') as csvfile:
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
                    imageToProcess = cv2.resize(imageToProcess, (5 * w, 5 * h), interpolation=cv2.INTER_LANCZOS4)
                    h, w = imageToProcess.shape[0], imageToProcess.shape[1]
                    print(h, w)
                    box = min(h, w)
                    handRectangles = [
                        # Left/Right hands person 0
                        [
                            op.Rectangle(100., 100., box, box),
                            op.Rectangle(0., 0., box, box),
                        ],

                        # [op.Rectangle(0, 0, w, w), op.Rectangle(0., 0., h, h)],
                        # Left/Right hands person 1
                        # [
                        # op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
                        # op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
                        # ],
                        # Left/Right hands person 2
                        # [
                        # op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
                        # op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
                        # ]
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

                    # frame = datum.cvOutputData
                    # cv2.rectangle(frame, (100, 100), (box, box), (0, 255, 0), 1, 1)
                    # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", frame)
                    # cv2.waitKey(0)
                    print(os.path.join(root_dir, gesture, folder, image), "working in progress.")
        print('FINISH THREAD {}'.format(self.gesture))

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
    # image_root_path = "D:\projects\openpose\examples\media/"
    # image_dir_list = os.listdir(image_root_path)
    # image_path = image_root_path + 'Gesture1/20/00010.jpg'
    # print('length is', len(image_dir_list))
    # for i in range(len(image_dir_list)):

    # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # parser.add_argument("--image_path", default=image_path,
    #                     help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # parser.add_argument("--body", default=0)
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

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # REPLACE THE root_dir W/ YOUR GESTURE DIR.
    # ALSO, IN "07hand_from_image.py", MAKE SURE TO *ONLY* PRINT "Hand" VARIABLE, WHICH IS A DICTIONARY.
    # 'Hand' should be a dictionary containing {'left': 2d array, 'right': 2d array}
    # root_dir = "D:\projects\openpose\examples\media/Gesture"
    # gestures = next(os.walk(root_dir))[1]
    # Hand = {}

    # 'field_names' are field names for the .csv file.
    # field_names = ['id', 'left', 'right']

    # START THREADS
    thread1 = gestureThread(0, gestures[0], 0)
    thread2 = gestureThread(1, gestures[1], 1)
    thread3 = gestureThread(2, gestures[2], 2)
    thread4 = gestureThread(3, gestures[3], 3)
    thread5 = gestureThread(4, gestures[4], 4)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()

    print('Exiting Main Thread')

    # for gesture_id, gesture in enumerate(gestures):
    #     folders = next(os.walk(root_dir + "/" + gesture))[1]
    #
    #     with open('results/result_' + str(gesture) + '.csv', 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=field_names)
    #         writer.writeheader()
    #         # Starting OpenPose
    #         opWrapper = op.WrapperPython()
    #         opWrapper.configure(params)
    #         opWrapper.start()
    #         for folder_id, folder in enumerate(folders):
    #             # x[1] denotes the files while scanning through the folder.
    #             images = [x[2] for x in os.walk(root_dir + "/" + gesture + "/" + folder)][0]
    #
    #             for image in images:
    #                 # print(os.path.join(root_dir, gesture, folder, image))
    #                 if image == '.DS_Store':
    #                     continue
    #
    #                 # # Starting OpenPose
    #                 # opWrapper = op.WrapperPython()
    #                 # opWrapper.configure(params)
    #                 # opWrapper.start()
    #
    #                 # Read image and face rectangle locations
    #                 image_path = os.path.join(root_dir, gesture, folder, image)
    #                 imageToProcess = cv2.imread(image_path)
    #                 h, w, c = imageToProcess.shape
    #                 new_size = int(max(h, w))
    #                 color = (0, 0, 0)
    #                 background = np.full((new_size, new_size, c), color, dtype=np.uint8)
    #                 ww = (new_size - w) // 2
    #                 hh = (new_size - h) // 2
    #                 background[hh:hh+h, ww:ww+w] = imageToProcess
    #                 imageToProcess = background
    #                 # imageToProcess = cv2.copyMakeBorder(imageToProcess, new_size-box, new_size-box, 0, 0, cv2.BORDER_CONSTANT)
    #                 # imageToProcess = cv2.resize(imageToProcess, (5*box, 5*box), interpolation = cv2.INTER_LANCZOS4)
    #                 h, w = imageToProcess.shape[0], imageToProcess.shape[1]
    #                 imageToProcess = cv2.resize(imageToProcess, (5 * w, 5 * h), interpolation=cv2.INTER_LANCZOS4)
    #                 h, w = imageToProcess.shape[0], imageToProcess.shape[1]
    #                 print(h, w)
    #                 box = min(h, w)
    #                 # imageToProcess = cv2.resize(imageToProcess, (w * 3, h * 3), interpolation=cv2.INTER_AREA)
    #                 handRectangles = [
    #                     # Left/Right hands person 0
    #                     [
    #                     op.Rectangle(100., 100., box, box),
    #                     op.Rectangle(0., 0., box, box),
    #                     ],
    #
    #                     # [op.Rectangle(0, 0, w, w), op.Rectangle(0., 0., h, h)],
    #                     # Left/Right hands person 1
    #                     # [
    #                     # op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
    #                     # op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
    #                     # ],
    #                     # Left/Right hands person 2
    #                     # [
    #                     # op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
    #                     # op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
    #                     # ]
    #                 ]
    #
    #                 # Create new datum
    #                 datum = op.Datum()
    #                 datum.cvInputData = imageToProcess
    #                 datum.handRectangles = handRectangles
    #
    #                 # Process and display image
    #                 opWrapper.emplaceAndPop([datum])
    #                 hand = {}
    #                 hand['id'] = folder+'_'+image[:-4]
    #                 left = str(datum.handKeypoints[0]).replace('\n', ',')
    #                 right = str(datum.handKeypoints[1]).replace('\n', ',')
    #                 # print("Left hand keypoints: \n" + left)
    #                 hand['left']=left
    #                 # print("Right hand keypoints: \n" + right)
    #                 hand['right']=right
    #                 # print('and Keypoints : \n', hand)
    #                 writer.writerow(hand)
    #                 print(hand)
    #
    #                 # with open(file_hands_path, 'w') as out:
    #                 #     # json.dump(hand_json, out)
    #                 #     # json.dump(hand, out)
    #                 #     out.write(str(hand))
    #                 # frame = datum.cvOutputData
    #                 # cv2.rectangle(frame, (100, 100), (box, box), (0, 255, 0), 1, 1)
    #                 # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", frame)
    #                 # cv2.waitKey(0)
    #                 print(os.path.join(root_dir, gesture, folder, image), "working in progress.")
except Exception as e:
    print(e)
    sys.exit(-1)