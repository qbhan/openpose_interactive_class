

def avg_hand_confidence(hand_keypoints):
    left_avg = sum(hand_keypoints[:, 2]) / 21
    return left_avg


def avg_pose_confidence(pose_keypoints):
    avg = sum(pose_keypoints[:, 2]) / 25
    return avg

'''Average of confidence of hand joints in the list'''
def avg_list_confidence(hand_list):
    all_avg = 0
    if len(hand_list) == 0:
        return 0
    for i in range(len(hand_list)):
        all_avg += avg_hand_confidence(hand_list[i])
    return all_avg / len(hand_list)


'''Check if the Hand joints are well recognized by Openpose through threshold'''
def valid_hand(prob, gesture_id):
    if prob is None or gesture_id is None:
        return False

    if prob < 0.3:
        return False

    return True


'''Calculate distance between two joints'''
def distance(prev, curr):
    return (prev[0] - curr[0]) ** 2 + (prev[1] - curr[1]) ** 2

'''Metric for detecting body movement using body joints'''
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

    '''
    Detect with following movements:
        1. nose and neck movements to detect turning head in a sudden
        2. shoulder movements to detect moving upper body in a sudden
        3. Center of the body to detect standing up&down 
    '''
    if distance_nose > 7000 and distance_neck > 6000:
        return True
    elif distance_left_shoulder > 4000 or distance_right_shoulder > 4000:
        return True
    elif distance_center > 5000:
        return True
    else:
        return False