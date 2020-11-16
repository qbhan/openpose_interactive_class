import sys
import cv2
import os
from sys import platform
import argparse

# prev_poseKeypoints =

def distance(prev, curr):
    return (prev[0] - curr[0]) ** 2 + (prev[1] - curr[1]) ** 2


def metric(pair_poseKeypoints):
    assert len(pair_poseKeypoints) == 2
    prev_poseKeypoints =  pair_poseKeypoints[0]
    poseKeypoints = pair_poseKeypoints[1]

    print('insie metric', type(prev_poseKeypoints))

    if len(prev_poseKeypoints) == 0 or len(poseKeypoints) == 0:
        return -1000
    print(len(prev_poseKeypoints), len(poseKeypoints))
    prev_nose, nose = prev_poseKeypoints[0], poseKeypoints[0]
    # print(prev_nose, nose)
    prev_neck, neck = prev_poseKeypoints[1], poseKeypoints[1]
    prev_right_shoulder, right_shoulder = prev_poseKeypoints[2], poseKeypoints[2]
    prev_left_shoulder, left_shoulder = prev_poseKeypoints[5], poseKeypoints[5]
    prev_right_eye, right_eye = prev_poseKeypoints[15], poseKeypoints[15]
    prev_left_eye, left_eye = prev_poseKeypoints[16], poseKeypoints[16]
    prev_right_ear, right_ear = prev_poseKeypoints[17], poseKeypoints[17]
    prev_left_ear, left_ear = prev_poseKeypoints[18], poseKeypoints[18]

    distance_nose = distance(prev_nose, nose)
    distance_neck = distance(prev_neck, neck)
    distance_right_shoulder = distance(prev_right_shoulder, right_shoulder)
    distance_left_shoulder = distance(prev_left_shoulder, left_shoulder)
    distance_right_eye = distance(prev_right_eye, right_eye)
    distance_left_eye = distance(prev_left_eye, left_eye)
    distance_right_ear = distance(prev_right_ear, right_ear)
    distance_left_ear = distance(prev_left_ear, left_ear)

    if distance_nose > 5000:
        return True
    elif distance_neck > 4000:
        return True
    else:
        return False


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
            # from openpose import *
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
            # from openpose import *
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    # params["net_resolution"] = "160x120"
    # params["hand"] = True
    # params["hand_detector"] = 2
    params["body"] = 1
    # params["face"] = True
    # params["disable_blending"] = True
    params["fps_max"] = 12

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # Fixed the handRectangles to only 1 person and 1 big rectangle, don't have to keep changing rectangle
    handRectangles = [[op.Rectangle(100.0, 150.0, 328.0, 328.0), op.Rectangle(0., 0., 0., 0.)]]
    # handRectangles = [[op.Rectangle(0.0, 0.0, 328.0, 328.0), op.Rectangle(0., 0., 0., 0.)]]

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    datum.handRectangles = handRectangles
    cam = cv2.VideoCapture(0)  # modify here for camera number
    # cam.set(3, 1280)
    # cam.set(4, 720)
    pair_poseKeypoints = [[], []]
    while (cv2.waitKey(1) != 27):
        # Get camera frame
        print('Another Frame')
        ret, frame = cam.read()
        # print(frame.shape[:2])
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        print('Here?')
        frame = datum.cvOutputData
        print(datum.poseKeypoints, datum.poseKeypoints.shape)
        # print(datum.poseKeypoints.shape == ())
        if datum.poseKeypoints.shape == ():
            cv2.imshow("Openpose 1.4.0 Webcam", frame)  # datum.cvOutputData
            continue
        # if
        print('new pose', len(datum.poseKeypoints[0]))
        del pair_poseKeypoints[0]
        print('Popped pair')
        pair_poseKeypoints.append(datum.poseKeypoints[0])
        print('Appended new pose')
        # print(len(pair_poseKeypoints))
        # if pair_poseKeypoints[0] != None:
        #     print(pair_poseKeypoints[0][0])
        # print(datum.poseKeypoints)
        # print(pair_poseKeypoints)
        moved = metric(pair_poseKeypoints)
        print(moved)

        # left_hand = datum.handKeypoints[0]
        # right_hand = datum.handKeypoints[1]
        # print(left_hand)
        if (moved):
            # cv2.rectangle(frame, (100, 150), (328, 328), (0, 255, 0), 1, 1)
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 20)
        cv2.imshow("Openpose 1.4.0 Webcam", frame)  # datum.cvOutputData
    # Always clean up
    cam.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(e)
    sys.exit(-1)
