# Reference Implementation: Intruder Detector

| Details           |          |
|-----------------------|-----------|
| Target OS:            |  Ubuntu\*   |
| Programming Language: |  C++      |
| Time to Complete:    |  50-70min |

![intruder-detector](../docs/images/intruder-detector-image.png)

An application capable of detecting any number of objects from a video input.

## What it Does
This application is one of a series of IoT reference implementations aimed at instructing users on how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. This solution detects any number of objects in a designated area, providing the number of objects in the frame and total count.

## How it Works
The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. A trained neural network detects objects within a designated area by displaying a green bounding box over them, and registers them in a logging system.

## Requirements
### Hardware
* 6th Generation Intel® Core™ processor with Intel® Iris® Pro graphics and Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)
*Note*: We recommend using a 4.14+ Linux* kernel with this software. Run the following command to determine your kernel version:

```
uname -a
```
* OpenCL™ Runtime Package
* OpenVINO™ toolkit

## Setup

### Install OpenVINO™ toolkit
Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux on how to install and setup the OpenVINO™ toolkit.

You will need the OpenCL™ Runtime Package if you plan to run inference on the GPU as shown by the instructions below. It is not mandatory for CPU inference.

### ffmpeg
ffmpeg is installed separately from the Ubuntu repositories:
```
sudo apt update
sudo apt install ffmpeg
```

## Build and start the application

To build, start a terminal in the `application` folder and run the following commands:

```
source env.sh
mkdir -p build && cd build
cmake ..
make
```
   
## Configure the application

### What model to use
The application works with any object-detection model, provided it has the same input and output format of the SSD model.  
The model can be any object detection model:
* that is provided by Intel® Distribution of OpenVINO™ toolkit.  
   These can be found in the `deployment_tools/intel_models` folder.
* downloaded using the **model downloader**, provided by Intel® Distribution of OpenVINO™ toolkit.   
   These can be found in the `deployment_tools/model_downloader/object_detection` folder.
* created by the user

By default this application uses the **person-vehicle-bike-detection-crossroad-0078** Intel® model, found in the `deployment_tools/intel_models` folder of the Intel® Distribution of OpenVINO™ toolkit installation.

### The labels file
In order to work, this application requires a _labels_ file associated with the model being used for detection.  
All detection models work with integer labels and not string labels (e.g. for the **person-vehicle-bike-detection-crossroad-0078** model, the number 1 represents the class "person"), that is why each model must have a _labels_ file, which associates an integer (the label the algorithm detects) with a string (denoting the human-readable label).   


The _labels_ file is a text file containing all the classes/labels that the model can recognize, in the order that it was trained to recognize them (one class per line). 
For the **person-vehicle-bike-detection-crossroad-0078** model, we provide the class file _labels.txt_ in the resources folder.


### The config file
The _resources/conf.txt_ contains the path to the videos that will be used by the application, followed by the labels to be detected on those videos. All labels (intruders) defined will be detected on all videos.   
The lines of the _conf.txt_ file are of the form `video: <path/to/video>` or `intruder: <label>`   
The labels used in the _conf.txt_ file must coincide with the labels from the _labels_ file.

Example of the _conf.txt_ file:
```
video: videos/video1.mp4
video: videos/video2.avi
intruder: person
intruder: dog
```

The application can use any number of videos for detection, but the more videos the application uses in parallel, the more the frame rate of each video scales down. This can be solved by adding more computation power to the machine the application is running on.


### What input video to use
The application works with any input video.
Sample videos for object detection are provided [here](https://github.com/intel-iot-devkit/sample-videos/).  


For first-use, we recommend using the [person-bicycle-car-detection]( https://github.com/intel-iot-devkit/sample-videos/blob/master/person-bicycle-car-detection.mp4) video.   
E.g.:
```
video: sample-videos/person-bicycle-car-detection.mp4
intruder: person
intruder: bicycle
intruder: car
```
This video can be downloaded directly, via the `video_downloader` python script provided. The script works with both python2 and python3. Run the following command:
```
python video_downloader.py
```
The video is automatically downloaded to the `application/resources/` folder.
   

### Using camera stream instead of video file
Replace `path/to/video` with the camera ID, where the ID is taken from yout video device (the number X in /dev/videoX).
On Ubuntu, to list all available video devices use the following command:
```
ls /dev/video*
```

## Run the application

If not in build folder, go there by using:

```
cd build/
```

To run the application with the needed models:

```
./intruder-detector -l ../resources/labels.txt -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.xml
```
   
### Having the input video loop
By default, the application reads the input videos only once, and ends when the videos ends.
In order to not have the sample videos end, thereby ending the application, the option to continously loop the videos is provided.    
This is done by running the application with the `-lp true` command-line argument:

```
./intruder-detector -lp true -l ../resources/labels.txt -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.xml
```

This looping does not affect live camera streams, as camera video streams are continuous and do not end.
   

## Running on different hardware
A user can specify what target device to run on by using the device command-line argument `-d` followed by one of the values `CPU`, `GPU`, or `MYRIAD`.   
If no target device is specified the application will default to running on the CPU.

### Running on the CPU
Although the application runs on the CPU by default, this can also be explicitly specified through the `-d CPU` command-line argument:
```
./intruder-detector -d CPU -l ../resources/labels.txt -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.xml
```
   

### Running on the integrated GPU
To run on the integrated Intel® GPU, use the `-d GPU` command-line argument:
```
./intruder-detector -d GPU -l ../resources/labels.txt -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.xml
```
### Running on the Intel® Neural Compute Stick
To run on the Intel® Neural Compute Stick, use the `-d MYRIAD` command-line argument:
```
./intruder-detector -d MYRIAD -l ../resources/labels.txt -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/person-vehicle-bike-detection-crossroad-0078/FP16/person-vehicle-bike-detection-crossroad-0078.xml
```
**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.
   

## Using the browser UI

The default application uses a simple user interface created with OpenCV.
A web based UI, with more features is also provided [here](../UI).
