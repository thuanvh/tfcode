"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from multiprocessing import Process, Queue

import numpy as np

import cv2

import sys
sys.path.insert(0,'thirdparty/yingoubing/head-pose-estimation')

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

CNN_INPUT_SIZE = 128


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main():
    """MAIN"""
    # Video source from webcam or video file.
    #video_src = "C:\\out\\20171114_104133.mp4"
    #cam = cv2.VideoCapture(video_src)
    #_, sample_frame = cam.read()
    txtfile = "biwi-all.txt"
    file_idx = 0
    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector(mark_model='thirdparty/yingoubing/head-pose-estimation/assets/frozen_inference_graph.pb', 
      dnn_proto_text='thirdparty/yingoubing/head-pose-estimation/assets/deploy.prototxt',
      dnn_model='thirdparty/yingoubing/head-pose-estimation/assets/res10_300x300_ssd_iter_140000.caffemodel')
    height = 480
    width = 640
    pose_estimator = PoseEstimator(img_size=(height, width), model_file='thirdparty/yingoubing/head-pose-estimation/assets/model.txt')
    with open(txtfile,"r") as ins:
      for line in ins:
        imgfile, labelfile = line.rstrip("\n").split(",")
        image = cv2.imread(imgfile)
        
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = image.shape[:2]      

        # Introduce scalar stabilizers for pose.
        pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]     

        facebox = mark_detector.extract_cnn_facebox(image)
        if facebox is not None:

          # Detect landmarks from image of 128x128.
          face_img = image[facebox[1]: facebox[3],
                          facebox[0]: facebox[2]]
          face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
          face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
          marks = mark_detector.detect_marks(face_img)

          # Convert the marks locations from local CNN to global image.
          marks *= (facebox[2] - facebox[0])
          marks[:, 0] += facebox[0]
          marks[:, 1] += facebox[1]

          # Uncomment following line to show raw marks.
          mark_detector.draw_marks(
              image, marks, color=(0, 255, 0))

          # Try pose estimation with 68 points.
          pose = pose_estimator.solve_pose_by_68_points(marks)
          #rotation_pose = pose[0].flatten()
          #print(rotation_pose)
          #rotCamerMatrix1 = cv2.Rodrigues(rotation_pose)
          pose_euler = pose_estimator.pose_angle(pose[0], pose[1]).flatten()
          pitch = -pose_euler[0]
          yaw = pose_euler[1]
          roll = pose_euler[2] - 180
          

          true_pose = np.genfromtxt(labelfile,delimiter=' ')
          true_pitch = -true_pose[0] #Biwi pose invert
          true_yaw = true_pose[1]
          true_roll = true_pose[2]

          print(true_yaw, true_pitch, true_roll, yaw, pitch, roll)
          # # Stabilize the pose.
          # stabile_pose = []
          # pose_np = np.array(pose).flatten()
          # for value, ps_stb in zip(pose_np, pose_stabilizers):
          #     ps_stb.update([value])
          #     stabile_pose.append(ps_stb.state[0])
          # stabile_pose = np.reshape(stabile_pose, (-1, 3))

          # # Uncomment following line to draw pose annotaion on frame.
          pose_estimator.draw_annotation_box(
              image, pose[0], pose[1], color=(255, 128, 128))

          # # Uncomment following line to draw stabile pose annotaion on frame.
          # pose_estimator.draw_annotation_box(
          #     image, stabile_pose[0], stabile_pose[1], color=(128, 255, 128))
          cv2.imshow("abc", image)
          if cv2.waitKey(10) == 27:
            break
     

if __name__ == '__main__':
    main()
