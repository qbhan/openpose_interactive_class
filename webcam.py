import sys
import cv2
import os
from sys import platform
import time

import torch
import torch.nn as nn

import http.client

conn = http.client.HTTPSConnection("api.zoom.us")
payload = "{\"message\":\"WELCOME TO THE MEETING\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_thumbsup = "{\"message\":\"NO PROBLEM\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_thumbsdown = "{\"message\":\"THUMBS DOWN\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_raisehand = "{\"message\":\"I HAVE A QUESTION!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_movement = "{\"message\":\"!!!!!!!!ALARM MOVEMENT DETECTED!!!!!!!!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_absent = "{\"message\":\"!!!!!!!!ABSENT!!!!!!!!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"

headers = {
    'content-type': "application/json",
    'authorization': "Bearer eyJhbGciOiJIUzUxMiIsInYiOiIyLjAiLCJraWQiOiI3OWFlYzUzMi0yYTAwLTQ3ODctOTIyNi00MmIxMGRlYjAzOTgifQ.eyJ2ZXIiOjcsImF1aWQiOiIzMjQ5ZjI1MDMwY2VhMDRjNTAwYTdkOGI1OTYzY2JiYSIsImNvZGUiOiJXZWp0ZmZWRnNhX2dzb2dMeTBiUm1lWkh1WTdzWHJ4MEEiLCJpc3MiOiJ6bTpjaWQ6NDF3Q3BackJUaWE1S0JCYlZNekFZUSIsImdubyI6MCwidHlwZSI6MCwidGlkIjowLCJhdWQiOiJodHRwczovL29hdXRoLnpvb20udXMiLCJ1aWQiOiJnc29nTHkwYlJtZVpIdVk3c1hyeDBBIiwibmJmIjoxNjA2MzkxNTY1LCJleHAiOjE2MDYzOTUxNjUsImlhdCI6MTYwNjM5MTU2NSwiYWlkIjoicVk3Y0FndWFTTEc1Sm13Skd1TnRjdyIsImp0aSI6IjQ5OTI0ODdjLTY0N2MtNDg3Mi1hMTVlLWVlM2I3NDRlNzIxOCJ9.ILvmW6GlcQwAcc0SSS5VN98kDHcYzAQB2bpo0TumhfO5f3Dk454q-xkJVRC-KVF4UA-0SBabseHAwyx0UfDs3A"
    }

conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload, headers)
#
res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


def avg_hand_confidence(hand_keypoints):
    left_avg = sum(hand_keypoints[:, 2]) / 21
    return left_avg


def avg_pose_confidence(pose_keypoints):
    avg = sum(pose_keypoints[:, 2]) / 25
    return avg


def avg_list_confidence(hand_list):
    all_avg = 0
    for i in range(len(hand_list)):
        all_avg += avg_hand_confidence(hand_list[i])
    return all_avg / len(hand_list)


def valid_hand(prob, gesture_id):
    if prob is None or gesture_id is None:
        return False

    if prob < 0.3:
        return False

    return True


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

    if len(prev_poseKeypoints) == 0 or len(poseKeypoints) == 0:
        return -1000
    prev_nose, nose = prev_poseKeypoints[0], poseKeypoints[0]
    prev_neck, neck = prev_poseKeypoints[1], poseKeypoints[1]
    prev_right_shoulder, right_shoulder = prev_poseKeypoints[2], poseKeypoints[2]
    prev_right_elbow, right_elbow = prev_poseKeypoints[3], poseKeypoints[3]
    prev_left_shoulder, left_shoulder = prev_poseKeypoints[5], poseKeypoints[5]
    prev_left_elbow, left_elbow = prev_poseKeypoints[6], poseKeypoints[6]
    prev_center, center = prev_poseKeypoints[6], poseKeypoints[6]
    prev_right_eye, right_eye = prev_poseKeypoints[15], poseKeypoints[15]
    prev_left_eye, left_eye = prev_poseKeypoints[16], poseKeypoints[16]
    prev_right_ear, right_ear = prev_poseKeypoints[17], poseKeypoints[17]
    prev_left_ear, left_ear = prev_poseKeypoints[18], poseKeypoints[18]

    distance_nose = distance(prev_nose, nose)
    distance_neck = distance(prev_neck, neck)
    distance_right_shoulder = distance(prev_right_shoulder, right_shoulder)
    distance_left_shoulder = distance(prev_left_shoulder, left_shoulder)
    distance_right_elbow = distance(prev_right_elbow, right_elbow)
    distance_left_elbow = distance(prev_left_elbow, left_elbow)
    distance_center = distance(prev_center, center)
    distance_right_eye = distance(prev_right_eye, right_eye)
    distance_left_eye = distance(prev_left_eye, left_eye)
    distance_right_ear = distance(prev_right_ear, right_ear)
    distance_left_ear = distance(prev_left_ear, left_ear)

    if distance_nose > 7000 and distance_neck > 6000:
        return True
    elif distance_left_elbow > 4000 or distance_right_elbow > 4000:
        return True
    elif distance_center > 5000:
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
        prev_state = None
        msg_state = ('not_sent', time.perf_counter())
        while (cv2.waitKey(1) != 27):
            if msg_state[0] == 'sent':
                if time.perf_counter() - msg_state[1] > 2.5:
                    msg_state = ('not_sent', time.perf_counter())


            ret, frame = cam.read()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            frame = datum.cvOutputData

            '''If Person not in Camera'''
            if datum.poseKeypoints.shape == ():
                # conn.request("POST", "/v2/chat/users/silvano211205@gmail.com/messages", payload, headers)
                #
                # res = conn.getresponse()
                # data = res.read()
                #
                # print(data.decode("utf-8"))
                if msg_state[0] == 'not_sent':
                    # print('WHY NOT WORKING')
                    conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload_absent, headers)
                    #
                    res = conn.getresponse()
                    data = res.read()

                    print(data.decode("utf-8"))
                    msg_state = ('sent', time.perf_counter())
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3
                fontColor = (255, 255, 0)
                lineType = 2
                fontThickness = 2
                msg_on_screen = 'ABSENT!'
                textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)

                cv2.rectangle(frame, (0, 0), (1280, 1024), (0, 0, 255), 20)
                cv2.putText(frame, 'Absent!',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                cv2.imshow("Openpose 1.4.0 Webcam", frame)
                continue

            '''Evaluate Movement & Confidence'''
            del pair_poseKeypoints[0]
            pair_poseKeypoints.append(datum.poseKeypoints[0])
            body_confidence_avg = avg_pose_confidence(datum.poseKeypoints[0])
            # print(body_confidence_avg)
            moved = metric(pair_poseKeypoints)

            '''Evaluate Hand Gesture'''
            if len(input_hands) == 12:
                del input_hands[0]
            input_hands.append(datum.handKeypoints[0][0])
            # print(len(input_hands))
            prob, gesture = None, None
            hand_confidence_avg = avg_list_confidence(input_hands)
            # if len(input_hands) == 12 and avg >= 0.1:
            if len(input_hands) == 12:
                # print('Confidence : ', hand_confidence_avg)
                prob, gesture = get_hand_gesture('normalizev2.pt', input_hands, 'cuda')
            # print(prob, gesture)



            '''Output Recognition Results'''
            print_msg = False
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = None
            fontColor = None
            fontScale = 3
            fontThickness = 2
            msg_on_screen = None

            if valid_hand(hand_confidence_avg, gesture) and gesture == 1:
                print('THUMBS DOWN PROB : ', prob)
                if prob > 11:
                    '''Counter'''

                    if prev_state is None:
                        prev_state = ('thumbs_down', time.perf_counter())
                        # print(prev_state)

                    elif prev_state[0] == 'rest':
                        if time.perf_counter() - prev_state[1] > 5.5:
                            prev_state = ('thumbs_down', time.perf_counter())
                            # print(prev_state)

                    elif prev_state[0] != 'thumbs_down':
                        prev_state = ('thumbs_down', time.perf_counter())
                        # print(prev_state)

                    else:
                        # print(time.perf_counter() - prev_state[1])
                        if msg_state[0] == 'not_sent':

                            conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload_thumbsdown,
                                         headers)
                            #
                            res = conn.getresponse()
                            data = res.read()

                            print(data.decode("utf-8"))
                            msg_state = ('sent', time.perf_counter())
                        if time.perf_counter() - prev_state[1] > 0.5:
                            print_msg = True
                            # bottomLeftCornerOfText = (450, 500)
                            fontColor = (255, 0, 0)
                            fontScale = 3
                            msg_on_screen = 'THUMBS DOWN'
                            textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                            bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)
                        # if time.perf_counter() - prev_state[1] > 3.5:
                        #     prev_state = ('rest', time.perf_counter())

            elif valid_hand(hand_confidence_avg, gesture) and gesture == 2:
                print('THUMBS DOWN PROB : ', prob)
                '''Counter'''
                if prob > 11:
                    if prev_state is None:
                        prev_state = ('thumbs up', time.perf_counter())
                        # print(prev_state)

                    elif prev_state[0] == 'rest':
                        if time.perf_counter() - prev_state[1] > 5.5:
                            prev_state = ('thumbs_up', time.perf_counter())
                            # print(prev_state)

                    elif prev_state[0] != 'thumbs_up':
                        prev_state = ('thumbs_up', time.perf_counter())
                        # print(prev_state)

                    else:
                        # print(time.perf_counter() - prev_state[1])
                        if msg_state[0] == 'not_sent':

                            conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload_thumbsup, headers)
                            #
                            res = conn.getresponse()
                            data = res.read()

                            print(data.decode("utf-8"))
                            msg_state = ('sent', time.perf_counter())
                        if time.perf_counter() - prev_state[1] > 0.5:

                            print_msg = True
                            # bottomLeftCornerOfText = (450, 500)
                            fontColor = (0, 255, 0)
                            fontScale = 3
                            msg_on_screen = 'THUMBS UP'
                            textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                            bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)
                        # if time.perf_counter() - prev_state[1] > 3.5:
                        #     prev_state = ('rest', time.perf_counter())


            elif valid_hand(hand_confidence_avg, gesture) and gesture == 4:
                print('RAISE HAND PROB : ', prob)
                '''Counter'''
                if prev_state is None:
                    prev_state = ('raise_hand', time.perf_counter())
                    # print(prev_state)

                elif prev_state[0] == 'rest':
                    if time.perf_counter() - prev_state[1] > 5.5:
                        prev_state = ('raise_hand', time.perf_counter())
                        # print(prev_state)

                elif prev_state[0] != 'raise_hand':
                    prev_state = ('raise_hand', time.perf_counter())
                    # print(prev_state)

                else:
                    # print(time.perf_counter() - prev_state[1])
                    if msg_state[0] == 'not_sent':

                        conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload_raisehand, headers)
                        #
                        res = conn.getresponse()
                        data = res.read()

                        print(data.decode("utf-8"))
                        msg_state = ('sent', time.perf_counter())
                    if time.perf_counter() - prev_state[1] > 0.5:
                        print_msg = True
                        bottomLeftCornerOfText = (450, 500)
                        fontColor = (0, 255, 255)
                        fontScale = 3
                        msg_on_screen = 'HAND RAISED'
                        textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                        bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)
                    # if time.perf_counter() - prev_state[1] > 3.5:
                    #     prev_state = ('rest', time.perf_counter())




            elif moved:

                '''Counter'''
                if prev_state is None:
                    prev_state = ('detect_move', time.perf_counter())
                    # print(prev_state)

                elif prev_state[0] == 'rest':
                    if time.perf_counter() - prev_state[1] > 1.5:
                        prev_state = ('detect_move', time.perf_counter())
                        # print(prev_state)

                elif prev_state[0] != 'detect_move':
                    prev_state = ('detect_move', time.perf_counter())
                    # print(prev_state)

                else:
                    # print(msg_state)
                    if msg_state[0] == 'not_sent':

                        conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload_movement, headers)
                        #
                        res = conn.getresponse()
                        data = res.read()

                        print(data.decode("utf-8"))
                        msg_state = ('sent', time.perf_counter())

                    print_msg = True
                    bottomLeftCornerOfText = (150, 500)
                    fontColor = (255, 255, 255)
                    fontScale = 3
                    msg_on_screen = 'MOVEMENT DETECTED'
                    textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                    # print(textsize)
                    bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)
                    # if time.perf_counter() - prev_state[1] > 3.5:
                    #     prev_state = ('rest', time.perf_counter())

            if print_msg:
                # print(data.decode("utf-8"))
                # bottomLeftCornerOfText = (550, 500)
                # fontScale = 2
                # fontColor = (255, 0, 0)
                lineType = 2
                cv2.rectangle(frame, (0, 0), (1280, 1024), fontColor, 40)
                cv2.putText(frame, msg_on_screen,
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