#!/usr/bin/python
import sys
import os.path
import subprocess
import ast
import csv
import threading

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

        with open('result' + gesture + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter='\t')
            writer.writeheader()

            for folder_id, folder in enumerate(folders):
                # x[1] denotes the files while scanning through the folder.
                images = [x[2] for x in os.walk(root_dir + "/" + gesture + "/" + folder)][0]

                for image in images:
                    print(os.path.join(root_dir, gesture, folder, image), '\n')
                    if image == '.DS_Store':
                        continue

# def run_thread_gesture(gesture):
#     print("got in")
#     folders = next(os.walk(root_dir + "/" + gesture))[1]
#
#     with open('result' + gesture + '.csv', 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter='\t')
#         writer.writeheader()
#
#         for folder_id, folder in enumerate(folders):
#             # x[1] denotes the files while scanning through the folder.
#             images = [x[2] for x in os.walk(root_dir + "/" + gesture + "/" + folder)][0]
#
#             for image in images:
#                 print(os.path.join(root_/dir, gesturrmfje, folder, image))
#                 if image == '.DS_Store':
#                     continue


if __name__ == '__main__':
    # **Implement the path manipulation.**
    # PLACE THIS .py FILE IN THE SAME FOLDER WHERE "07hand_from_image.py" IS PLACED.
    # REPLACE THE root_dir W/ YOUR GESTURE DIR.
    # ALSO, IN "07hand_from_image.py", MAKE SURE TO *ONLY* PRINT "Hand" VARIABLE, WHICH IS A DICTIONARY.
    # 'Hand' should be a dictionary containing {'left': 2d array, 'right': 2d array}
    root_dir = "D:\projects\openpose\examples\media/Gesture"
    gestures = next(os.walk(root_dir))[1]
    Hand = {}
    # 'field_names' are field names for the .csv file.
    field_names = ['id', 'left', 'right']
    print(gestures, type(gestures[1]))
    # each 'gesture' and 'image' is a path to the gesture folders and the image files respectively.
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
    #     with open('result' + str(gesture_id) + '.csv', 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter='\t')
    #         writer.writeheader()
    #
    #         for folder_id, folder in enumerate(folders):
    #             # x[1] denotes the files while scanning through the folder.
    #             images = [x[2] for x in os.walk(root_dir + "/" + gesture + "/" + folder)][0]
    #
    #             for image in images:
    #                 # print(os.path.join(root_/dir, gesturrmfje, folder, image))
    #                 if image == '.DS_Store':
    #                     continue

                    # print(folder+'_'+image[:-4])
                    # proc = subprocess.Popen(["python3 07hand_from_image.py --image_path=" + image], stdout=subprocess.PIPE,
                    #                         shell=True)
                    #
                    # # 'out' is a bytes type variable with involves a "Hand" dictionary.
                    # (out, err) = proc.communicate()
                    #
                    # # 'Hand' is a dictionary with keys: ['id', 'left', 'right'], and values: [path, (21x3) 2d array, (21x3) 2d array].
                    # Hand = out.decode("UTF-8")
                    # Hand = ast.literal_eval(Hand)
                    # Hand['id'] = root_dir + "-" + str(gesture_id) + "-" + str(folder_id) + "-" + image
                    #
                    # writer.writerow(Hand)
                    # print(root_dir + "/" + gesture + "/" + image, "working in progress.")