from face_detection import FaceDetectionModel
from facial_landmarks_detection import FaceLandmarksDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel

from mouse_controller import MouseController
from input_feeder import InputFeeder

from argparse import ArgumentParser
import os
import mimetypes
import cv2
import time
import numpy as np

def build_argparser():
    """
    Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-f", "--face_model", required=True, type=str, help="Path to face detection model")
    parser.add_argument("-l", "--face_landmarks_model", required=True, type=str, help="Path to face landmark detection model")
    parser.add_argument("-p", "--head_pose_model", required=True, type=str, help="Path to head pose estimation model")
    parser.add_argument("-g", "--gaze_model", required=True, type=str, help="Path to gaze estimation model")
    parser.add_argument("-i", "--input", default="cam", type=str, help="Path to video file or default is set to cam which takes frames from webcam")
    parser.add_argument("-d", "--device", default="CPU", type=str, help="Provide target device like CPU, GPU, FPGA, MYRIAD")
    parser.add_argument("-e", "--extension", type=str, default=None, help="Provice custom layers extensions")

    return parser

def get_file_type(input_file):
    abs_path = os.path.abspath(input_file)
    mime_type = mimetypes.guess_type(abs_path)
    if "image" in mime_type[0]:
        return "image"
    elif "video" in mime_type[0]:
        return "video"
    else:
        return False

def draw_box(coords, image):
        '''
        Draw bounding box for input image
        '''
        image = cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 255), 2)
        return image

def main():
    args = build_argparser().parse_args()

    if args.input == "cam":
        feed = InputFeeder(args.input)
    else:
        file_type = get_file_type(args.input)
        if file_type:
            feed = InputFeeder(file_type, args.input)
        else: 
            print("File type not supported")
            exit(1)

    # Load all the models
    try:
        start_time = time.time()
        face_detection_model = FaceDetectionModel(args.face_model, args.device, args.extension)
        total_time = time.time() - start_time
        print(f"Face Detection model took {total_time:.3f}s to load")
        start_time = time.time()
        face_landmark_detection_model = FaceLandmarksDetectionModel(args.face_landmarks_model, args.device, args.extension)
        total_time = time.time() - start_time
        print(f"Face Landmarks detection model took {total_time:.3f}s to load")
        start_time = time.time()
        head_pose_estimation_model = HeadPoseEstimationModel(args.head_pose_model, args.device, args.extension)
        total_time = time.time() - start_time
        print(f"Head pose estimation model took {total_time:.3f}s to load")
        start_time = time.time()
        gaze_estimation_model = GazeEstimationModel(args.gaze_model, args.device, args.extension)
        total_time = time.time() - start_time
        print(f"Gaze estimation model took {total_time:.3f}s to load")
        print("Models loaded successfully...!!!")
    except Exception as e:
        print("Model failed to load", e)
    
    mouse_controller = MouseController("high", "fast")

    feed.load_data()

    # benchmarking
    frame_count = 0
    inference_times_frames = []
    face_infer_times = []
    land_infer_times = []
    pose_infer_times = []
    gaze_infer_times = []
    start_total_inf_time = time.time()
    try:
        for batch in feed.next_batch():
            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                feed.close()
                break
            if batch is None:
                feed.close()
                break
            vis_image = batch.copy()
            start_frame_inference = time.time()
            try:
                start_time = time.time()
                face_crop, box = face_detection_model.predict(batch.copy())
                total_time = time.time() - start_time
                face_infer_times.append(total_time) 
                vis_image = draw_box(box, vis_image)
                if len(face_crop) == 0:
                    print("No face detected in the frame")
                    continue
            except Exception as e:
                print("Error in face detection inference", e)
                exit(1)
            
            try:
                start_time = time.time()
                l_eye_img, r_eye_img, eye_coordinates = face_landmark_detection_model.predict(face_crop.copy())
                total_time = time.time() - start_time
                land_infer_times.append(total_time)
                left_eye_mask = draw_box(eye_coordinates[0], face_crop.copy())
                eyes_masked = draw_box(eye_coordinates[1], left_eye_mask)
                vis_image[box[1]:box[3], box[0]:box[2]] = eyes_masked
            except Exception as e:
                print("Error in landmark detection inference", e)
                exit(1)

            try:
                start_time = time.time()
                head_ypr_angle = head_pose_estimation_model.predict(face_crop.copy())
                total_time = time.time() - start_time
                pose_infer_times.append(total_time)
            except Exception as e:
                print("Error in head pose estimation inference", e)
                exit(1)

            try:
                start_time = time.time()
                mouse_x, mouse_y = gaze_estimation_model.predict(l_eye_img.copy(), r_eye_img.copy(), head_ypr_angle)
                total_time = time.time() - start_time
                gaze_infer_times.append(total_time)
            except Exception as e:
                print("Error in gaze estimation inference", e)
                exit(1)
            frame_count += 1
            total_inference_time = time.time() - start_frame_inference
            inference_times_frames.append(total_inference_time)
            cv2.imshow('visualization', cv2.resize(vis_image,(720,720)))
            mouse_controller.move(mouse_x, mouse_y)
        feed.close()

        total_time_for_inference = time.time() - start_total_inf_time

        print(f"Total number of frames processed - {frame_count}")
        print(f"FPS {frame_count / total_time_for_inference}")
        print(f"Average time it took to process each frame: {np.mean(inference_times_frames):.3f}s")
        print(f"Average time it took for processing face detection: {np.mean(face_infer_times):.3f}s")
        print(f"Average time it took for processing face landmark detection: {np.mean(land_infer_times):.3f}s")
        print(f"Average time it took for processing head pose estimation: {np.mean(pose_infer_times):.3f}s")
        print(f"Average time it took for processing gaze estimation: {np.mean(gaze_infer_times):.3f}s")
    except Exception as e:
        print("Error occured", e)
        exit(1)
if __name__ == "__main__":
    main()