import sys
import cv2
import os
from sys import platform
import argparse

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
    params["face"] = True
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
    cam.set(3, 1280)
    cam.set(4, 720)
    while (cv2.waitKey(1) != 27):
        # Get camera frame
        ret, frame = cam.read()
        # print(frame.shape[:2])
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        frame = datum.cvOutputData
        print(datum.poseKeypoints)
        # left_hand = datum.handKeypoints[0]
        # right_hand = datum.handKeypoints[1]
        # print(left_hand)
        # cv2.rectangle(frame, (100, 150), (328, 328), (0, 255, 0), 1, 1)
        # cv2.rectangle(frame, (0, 0), (328, 328), (0, 255, 0), 1, 1)
        cv2.imshow("Openpose 1.4.0 Webcam", frame)  # datum.cvOutputData
    # Always clean up
    cam.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(e)
    sys.exit(-1)
