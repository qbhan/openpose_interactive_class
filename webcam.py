import sys
import cv2
import os
from sys import platform
import argparse

import torch
import torch.nn as nn


def avg_confidence(hand_keypoints):
    left_avg = sum(hand_keypoints[:, 2]) / 21
    return left_avg


def avg_list_confidence(hand_list):
    all_avg = 0
    for i in range(len(hand_list)):
        all_avg += avg_confidence(hand_list[i])
    return all_avg / len(hand_list)

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        output = self.act(self.bn1(self.fc1(x)))
        # output = self.act(self.fc1(x))
        output = self.act(self.bn2(self.fc2(output)))
        # output = self.act(self.fc2(output))
        output = self.act(self.bn3(self.fc3(output)))
        # output = self.act(self.fc3(output))
        return output


class GestureDetector(nn.Module):
    def __init__(self, nf):
          super(GestureDetector, self).__init__()
          block = MLPBlock
          self.mlp = block(12*21*3, nf)
          self.fc = nn.Linear(nf, 5) # our gesture dataset is consisted of 5 classes

    def forward(self, x):
        # print(x.view(x.size()[0], -1).shape)
        # print(torch.flatten(torch.flatten(x, start_dim=1), start_dim=0).shape)
        output = self.mlp(x.view(x.size()[0], -1))
        output = self.fc(output)
        return output


def get_hand_gesture(model_name, input, device):
    model = GestureDetector(64).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    input = torch.FloatTensor(input).to(device)
    input = input.unsqueeze(0)
    # print(input.shape)
    output = model(input)
    # print(output.shape)
    pred = output.argmax(dim=1).data[0]
    # print(pred == 3)
    prob = output[0][pred].data
    return prob, pred


def distance(prev, curr):
    return (prev[0] - curr[0]) ** 2 + (prev[1] - curr[1]) ** 2


def metric(pair_poseKeypoints):
    assert len(pair_poseKeypoints) == 2
    prev_poseKeypoints =  pair_poseKeypoints[0]
    poseKeypoints = pair_poseKeypoints[1]

    # print('insie metric', type(prev_poseKeypoints))

    if len(prev_poseKeypoints) == 0 or len(poseKeypoints) == 0:
        return -1000
    # print(len(prev_poseKeypoints), len(poseKeypoints))
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


def main():
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

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"
        # params["net_resolution"] = "1920x1080"
        params["hand"] = True
        params["hand_detector"] = 2
        params["body"] = 1
        params["net_resolution"] = '320x192'  #20*11
        # params["face"] = True
        # params["disable_blending"] = True
        # params["fps_max"] = 5
        handRectangles = [[op.Rectangle(128, 0, 1024, 1024), op.Rectangle(0., 0., 0., 0.)]]
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        datum.handRectangles = handRectangles
        cam = cv2.VideoCapture(0)  # modify here for camera number
        cam.set(3, 1280)
        cam.set(4, 1024)
        pair_poseKeypoints = [[], []]
        input_hands = []
        while (cv2.waitKey(1) != 27):
            # Get camera frame
            # print('Another Frame')
            ret, frame = cam.read()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            frame = datum.cvOutputData

            if datum.poseKeypoints.shape == ():
                cv2.imshow("Openpose 1.4.0 Webcam", frame)  # datum.cvOutputData
                continue
            del pair_poseKeypoints[0]
            pair_poseKeypoints.append(datum.poseKeypoints[0])
            moved = metric(pair_poseKeypoints)
            # print(moved)

            # input hand gesture
            # assert len(input_hands) > 12
            # confidence = avg_confidence(datum.handKeypoints[0][0])
            # print('Confidence : ', confidence)
            # if confidence > 0.3:
            if len(input_hands) == 12:
                del input_hands[0]
            input_hands.append(datum.handKeypoints[0][0])
            # print(len(input_hands))
            prob, gesture = None, None
            avg = avg_list_confidence(input_hands)
            # if len(input_hands) == 12 and avg >= 0.1:
            if len(input_hands) == 12:
                print('Confidence : ', avg)
                prob, gesture = get_hand_gesture('normalizev2.pt', input_hands, 'cuda')
            print(prob, gesture)

            if moved:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 500)
                fontScale = 1
                fontColor = (0, 0, 0)
                lineType = 2
                cv2.rectangle(frame, (0, 0), (1280, 1024), (0, 0, 255), 20)
                cv2.putText(frame, 'Hello World!',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
            cv2.imshow("Openpose 1.4.0 Webcam", frame)  # datum.cvOutputData
        # Always clean up
        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        sys.exit(-1)

if __name__ == '__main__':
    main()