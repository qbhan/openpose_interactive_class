# Any Questions? 🖐🖐🖐🖐🖐
## Motion detector for interactive online real-time class
We deliver a body movement & hand gesture detector that can enhance the interactivity of online real-time classes.
## Demo
gif of my face
youtube link

## Quantitative Results
<p align="center"><img src="static/MLP_mid12.png" width="33%"><img src="static/MLP_mid2.png" width="33%"><img src="static/MLP_max2.png" width="33%"></p>
<p align="center"><img src="static/CNN_mid12.png" width="33%"><img src="static/CNN_duplicate100.png" width="33%">
</p>
Confusion matrices of each model. The upper line shows the results of the MLP classifier and the lower line shows the results of the CNN-based classifier. The column names indicate predicted labels and the row names indicate the ground truth labels.


## Methods
### Flow of application
<p align="center"><img src="static/application.PNG" width=80%></p>

Process of our application. Motion detection and gesture recognition are independent of each other. Both tasks use OpenPose to get the keypoints. Motion detection is based on pre-defined metrics which are manually designed. Gesture recognition is done by a classifier. We implemented two different classifiers.

### Data preprocessing
To improve the training quality, we first preprocessed the dataset in two ways: normalizing and modifying the length of sequence.
#### 1. Normalizing
We normalized the x, y coordinates of the hand keypoints in each frame to force the model to learn only the gesture, not the location of hand on the image. Normalization process also allows the model to be applied regardless of the resolution of the video.
#### 2. Modifying the length of sequence
Since the two classifiers take only fixed-size inputs, we modified the length of sequence using three different metrics: *Mid*, *MaxConf*, and *Duplicate*.

+ *Mid* extracts a fixed number of frames in the 60% point of the full sequence. For example, if we are planning to use 12 from the 20 frames, we took the 6th~17th.

+ *MaxConf* uses the frame with maximum confidence and its neighbors. Frame with high confidence is likely to contain our target gesture because the gestures are static and done right in front of the camera, which makes OpenPose easily find the joints.

+ *Duplicate* matches the length by duplicating the frames. For example, when the original sequence is {1, 2, 3} and we want to make the length to 10, it becomes {1, 1, 1, 2, 2, 2, 2, 3, 3, 3}.

### MLP classifier
Considering the real-time setting and relatively small input size, we decided to use a small MLP classifier. It takes a sequence of hand keypoints with the confidence level for each joint and predicts the gesture. However, a simple MLP classifier cannot learn any temporal information, which is crucial for gesture recognition. Check the MLP model training code [here (GestureMLP.ipynb)](https://colab.research.google.com/drive/1amDIhHZz_WtkFU0zPwo986QEVdvbEX_S#scrollTo=YmFlCTA0u-4W&uniqifier=2).

### CNN-based classifier

<p align="center"><img src="static/CNN_classifier.PNG" width=80%></p>

We also trained a CNN-based classifier to overcome the disadvantages of the MLP classifier. We used the model architecture from the paper [Deep Learning for Hand Gesture Recognition on Skeletal Data](https://ieeexplore.ieee.org/document/8373818). In this work, they used multi-channel CNN to extract channel-specific features. The channel indicates each coordinate (x, y) of a joint. Each CNN consists of two feature extractors and one residual branch. They produces channel-specific features by concatenating the three outputs. Final MLP layer takes the extracted features as input and predicts the gesture. We modified some minor points of the existing model, such as input/output dimensions and activation function. Check the CNN-based model training code [here (GestureCNN.ipynb)](https://colab.research.google.com/drive/1EgJt0P3w28_fkQxq__0R_s_88VGasuGg#scrollTo=jTqC9q7HPVno).

## Required Installation
* Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* pytorch (>= 1.6.0 to use model trained on Colab default settings in local environment)
* npm (for Zoom SDK)
Caution about Zoom

## Reference
