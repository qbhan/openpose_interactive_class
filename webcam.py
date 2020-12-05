import sys
import cv2
import os
from sys import platform
import time

# import torch
# import torch.nn as nn

from model import *
from metrics import *

import http.client

conn = http.client.HTTPSConnection("api.zoom.us")
payload = "{\"message\":\"ATTENDENCE CHECK\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_thumbsup = "{\"message\":\"NO PROBLEM\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_thumbsdown = "{\"message\":\"THUMBS DOWN\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_raisehand = "{\"message\":\"I HAVE A QUESTION!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_movement = "{\"message\":\"!!!!!!!!ALARM MOVEMENT DETECTED!!!!!!!!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_absent = "{\"message\":\"!!!!!!!!ABSENT!!!!!!!!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"
payload_multi = "{\"message\":\"!!!!!!!!MULTIPLE PEOPLE!!!!!!!!\",\"to_channel\":\"c18c9ce3-6017-4f68-9ead-bb938eb565af\"}"

headers = {
    'content-type': "application/json",
    'authorization': "Bearer eyJhbGciOiJIUzUxMiIsInYiOiIyLjAiLCJraWQiOiI2ZDgzNmU5NC1mOTU2LTQ4N2UtYjZiMS01Y2Q1MzJmNTIwMjgifQ.eyJ2ZXIiOjcsImF1aWQiOiIzMjQ5ZjI1MDMwY2VhMDRjNTAwYTdkOGI1OTYzY2JiYSIsImNvZGUiOiIxTEhKbG1iVHZ4X2dzb2dMeTBiUm1lWkh1WTdzWHJ4MEEiLCJpc3MiOiJ6bTpjaWQ6NDF3Q3BackJUaWE1S0JCYlZNekFZUSIsImdubyI6MCwidHlwZSI6MCwidGlkIjowLCJhdWQiOiJodHRwczovL29hdXRoLnpvb20udXMiLCJ1aWQiOiJnc29nTHkwYlJtZVpIdVk3c1hyeDBBIiwibmJmIjoxNjA2NDQzMjgwLCJleHAiOjE2MDY0NDY4ODAsImlhdCI6MTYwNjQ0MzI4MCwiYWlkIjoicVk3Y0FndWFTTEc1Sm13Skd1TnRjdyIsImp0aSI6ImZkYWFiOWUzLWZhZTktNDI3My04MDI1LTU0YTg5NDA1NDM2ZiJ9.ms3Jd0tn_a7MXEXSbOSsBMf6LinTraLtA7g0hGBJLpDzWXzihEnTQ1C06HxmgXrn4cZ_Yr7TZoUDl2GdkuiPZQ"
    }

conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload, headers)
#
res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


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

        model = GestureDetector(frames=12, nf=64).to('cuda')
        model.load_state_dict(torch.load('normalizev2.pt'))
        model.eval()

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
                fontColor = (0, 0, 255)
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

            if len(datum.poseKeypoints) > 1:
                if prev_state is not None and prev_state[0] == 'multi_people':
                    if prev_state[1] > 2:
                        if msg_state[0] == 'not_sent':
                            # print('WHY NOT WORKING')
                            conn.request("POST", "/v2/chat/users/smarthkb98@kaist.ac.kr/messages", payload_multi, headers)
                            #
                            res = conn.getresponse()
                            data = res.read()

                            print(data.decode("utf-8"))
                            msg_state = ('sent', time.perf_counter())
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 3
                        fontColor = (0, 0, 255)
                        lineType = 2
                        fontThickness = 2
                        msg_on_screen = 'MUTLIPLE PEOPLE!'
                        textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                        bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)

                        cv2.rectangle(frame, (0, 0), (1280, 1024), (0, 0, 255), 20)
                        cv2.putText(frame, msg_on_screen,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        cv2.imshow("Openpose 1.4.0 Webcam", frame)
                        msg_state = ('sent', time.perf_counter())
                        continue
                else:
                    prev_state = ('multi_people', time.perf_counter())
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
                prob, gesture = get_hand_gesture(model, input_hands)
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
                print('THUMBS UP PROB : ', prob)
                '''Counter'''
                if prob > 14:
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
                    fontColor = (0, 0, 255)
                    fontScale = 3
                    msg_on_screen = 'MOVEMENT DETECTED'
                    textsize = cv2.getTextSize(msg_on_screen, font, fontScale, fontThickness)[0]
                    # print(textsize)
                    bottomLeftCornerOfText = ((1280 - textsize[0]) // 2, (1024 + textsize[1]) // 2)
                    # if time.perf_counter() - prev_state[1] > 3.5:
                    #     prev_state = ('rest', time.perf_counter())

            if print_msg:
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