# Computer Pointer Controller

Computer pointer controller is basically controlling your pointer device i.e., mouse based on the eye glaze. We can use video file, image or webcam stream as input
It uses 4 different models to analyze where to move the pointer.

## Project Set Up and Installation

### Prerequisites

- Python >= 3.6

- OpenVINO >=2020.2

- OpenCV

### Steps to install

#### Step 1

Clone this repository

#### Step 2 (optional)

Create a virtual environment

#### Step 3

Install requirements for the project

```command
pip install pyautogui
sudo apt-get install python3-tk python3-dev
```

#### Step 4

Initialize openvino environment variables it will vary based on OS below is for linux

```command
source /opt/intel/openvino/bin/setupvars.sh
```

#### Step 5

Download models required for this project. Model downloader command varies based on OS. Below are the commands for linux
Run below commands from the root path of the project to download the models in same directory

- [Face detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html))
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html))
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html))

```command
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

## Demo

Following is the command to run the project demo which uses demo video

```command
python ./src/main.py -i ./bin/demo.mp4 \
-f intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
-l intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
-p intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
-g intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
```

## Documentation

This project main file supports multiple arguments

```command
usage: main.py [-h] -f FACE_MODEL -l FACE_LANDMARKS_MODEL -p HEAD_POSE_MODEL
               -g GAZE_MODEL [-i INPUT] [-d DEVICE] [-e EXTENSION]

optional arguments:
  -h, --help            show this help message and exit
  -f FACE_MODEL, --face_model FACE_MODEL
                        Path to face detection model
  -l FACE_LANDMARKS_MODEL, --face_landmarks_model FACE_LANDMARKS_MODEL
                        Path to face landmark detection model
  -p HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                        Path to head pose estimation model
  -g GAZE_MODEL, --gaze_model GAZE_MODEL
                        Path to gaze estimation model
  -i INPUT, --input INPUT
                        Path to video file or default is set to cam which
                        takes frames from webcam
  -d DEVICE, --device DEVICE
                        Provide target device like CPU, GPU, FPGA, MYRIAD
  -e EXTENSION, --extension EXTENSION
                        Provice custom layers extensions
```

### Project Structure

```text
env
intel
    intel
    │   ├── face-detection-adas-binary-0001
    │   │   └── FP32-INT1
    │   │       ├── face-detection-adas-binary-0001.bin
    │   │       └── face-detection-adas-binary-0001.xml
    │   ├── gaze-estimation-adas-0002
    │   │   ├── FP16
    │   │   │   ├── gaze-estimation-adas-0002.bin
    │   │   │   └── gaze-estimation-adas-0002.xml
    │   │   ├── FP16-INT8
    │   │   │   ├── gaze-estimation-adas-0002.bin
    │   │   │   └── gaze-estimation-adas-0002.xml
    │   │   └── FP32
    │   │       ├── gaze-estimation-adas-0002.bin
    │   │       └── gaze-estimation-adas-0002.xml
    │   ├── head-pose-estimation-adas-0001
    │   │   ├── FP16
    │   │   │   ├── head-pose-estimation-adas-0001.bin
    │   │   │   └── head-pose-estimation-adas-0001.xml
    │   │   ├── FP16-INT8
    │   │   │   ├── head-pose-estimation-adas-0001.bin
    │   │   │   └── head-pose-estimation-adas-0001.xml
    │   │   └── FP32
    │   │       ├── head-pose-estimation-adas-0001.bin
    │   │       └── head-pose-estimation-adas-0001.xml
    │   └── landmarks-regression-retail-0009
    │       ├── FP16
    │       │   ├── landmarks-regression-retail-0009.bin
    │       │   └── landmarks-regression-retail-0009.xml
    │       ├── FP16-INT8
    │       │   ├── landmarks-regression-retail-0009.bin
    │       │   └── landmarks-regression-retail-0009.xml
    │       └── FP32
    │           ├── landmarks-regression-retail-0009.bin
    │           └── landmarks-regression-retail-0009.xml
app
    ├── bin
    │   └── demo.mp4
    ├── README.md
    ├── requirements.txt
    └── src
        ├── face_detection.py
        ├── facial_landmarks_detection.py
        ├── gaze_estimation.py
        ├── head_pose_estimation.py
        ├── input_feeder.py
        ├── main.py
        ├── mouse_controller.py
```

## Benchmarks

Hardware used for running inference

```text
product: Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz
size: 998MHz
capacity: 3500MHz
width: 64 bits
ram: 8GB
```

### Scenario 1

Face Detection: FP32-INT1
Face Landmarks Detection: FP32
Head Pose Estimation: FP32
Gaze Estimation: FP32

```text
Face Detection model took 0.158s to load
Face Landmarks detection model took 0.068s to load
Head pose estimation model took 0.066s to load
Gaze estimation model took 0.111s to load
Models loaded successfully...!!!
Total number of frames processed - 59
FPS 0.731892829513098
Average time it took to process each frame: 0.040s
Average time it took for processing face detection: 0.033s
Average time it took for processing face landmark detection: 0.002s
Average time it took for processing head pose estimation: 0.003s
Average time it took for processing gaze estimation: 0.002s
```

### Scenario 2

Face Detection: FP32-INT1
Face Landmarks Detection: FP16
Head Pose Estimation: FP16
Gaze Estimation: FP32

```text
Face Detection model took 0.117s to load
Face Landmarks detection model took 0.124s to load
Head pose estimation model took 0.197s to load
Gaze estimation model took 0.091s to load
Models loaded successfully...!!!
Total number of frames processed - 59
FPS 0.7350023101296336
Average time it took to process each frame: 0.037s
Average time it took for processing face detection: 0.030s
Average time it took for processing face landmark detection: 0.001s
Average time it took for processing head pose estimation: 0.003s
Average time it took for processing gaze estimation: 0.002s
```

### Scenario 3

Face Detection: FP32-INT1
Face Landmarks Detection: FP16
Head Pose Estimation: FP16
Gaze Estimation: FP16

```text
Face Detection model took 0.396s to load
Face Landmarks detection model took 0.134s to load
Head pose estimation model took 0.207s to load
Gaze estimation model took 0.208s to load
Models loaded successfully...!!!
Total number of frames processed - 59
FPS 0.7379265135704528
Average time it took to process each frame: 0.038s
Average time it took for processing face detection: 0.031s
Average time it took for processing face landmark detection: 0.001s
Average time it took for processing head pose estimation: 0.002s
Average time it took for processing gaze estimation: 0.003s
```

### Scenario 4

Face Detection: FP32-INT1
Face Landmarks Detection: FP16-INT8
Head Pose Estimation: FP16-INT8
Gaze Estimation: FP32-INT8

```text
Face Detection model took 0.239s to load
Face Landmarks detection model took 0.364s to load
Head pose estimation model took 0.233s to load
Gaze estimation model took 0.243s to load
Models loaded successfully...!!!
Total number of frames processed - 59
FPS 0.7364297369192886
Average time it took to process each frame: 0.037s
Average time it took for processing face detection: 0.032s
Average time it took for processing face landmark detection: 0.001s
Average time it took for processing head pose estimation: 0.002s
Average time it took for processing gaze estimation: 0.001s
```

## Results

In the above scenarios we can see that Face landmarks detection and headpose estimation models are loading faster with FP32 precision.
But we can see that FP32 models are bit faster in loading to CPU and INT8 is taking more time load

When we see inference time FP16 are runnig faster compared to FP32 models.

## Stand Out Suggestions

- Running in different precisions gives better results like FP-16 are faster compared to FP-32

- Added benchmark of model load time and inference time by default

### Edge Cases

- In the darker light conditions model was not able to detect eye glaze and mouse pointer starts moving randomly

- Also if the camera quality is low model will give poor results when I tried with my laptop camera it was not great but when I tried with external webcam
it works well
