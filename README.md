# Towards Markerless Surgical Tool and Hand Pose Estimation: Real Dataset Generation

- [Project page](http://medicalaugmentedreality.org/handobject.html)
- [Synthetic Grasp Generation](https://github.com/jonashein/grasp_generator)
- [Synthetic Grasp Rendering](https://github.com/jonashein/grasp_renderer)
- [Real Dataset Generation](https://github.com/jonashein/handobject_dataset_creator)
- [HandObjectNet Baseline](https://github.com/jonashein/handobjectnet_baseline)
- [PVNet Baseline](https://github.com/jonashein/pvnet_baseline)
- [Combined Model Baseline](https://github.com/jonashein/baseline_combination)

Our real dataset is available on the [project page](http://medicalaugmentedreality.org/handobject.html).

<!-- - [Paper](http://arxiv.org/abs/2004.13449) -->

## Table of Content

- [Setup](#setup)
- [Recording Data](#recording-data)
- [Camera Setup and Calibration](#camera-setup-and-calibration)
- [Ground Truth Recovery](#ground-truth-recovery)
- [Citations](#citations)

## Setup

Retrieve the code
```sh
git clone https://github.com/jonashein/handobject_dataset_recorder.git
cd handobject_dataset_recorder
```

### Prerequisites

Install [OpenCV](https://opencv.org/), [Eigen](https://eigen.tuxfamily.org/), and [PCL](https://pointclouds.org/):
```sh
sudo apt-get install libopencv-dev libeigen3-dev libpcl-dev
```

#### Azure Kinect SDK

Install the [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#debian-package). 
Make sure to follow the [Linux device setup](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#linux-device-setup) steps.

```sh
sudo apt-get install libk4a1.4 libk4a1.4-dev k4a-tools
```

### Download the MANO Model Files

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the content of the *models* folder into the `assets/mano` folder

- Your structure should look like this:

```
handobject_dataset_recorder/
  assets/
    mano/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      fhb_skel_centeridx0.pkl
      fhb_skel_centeridx9.pkl
```

### Install Code

Create and activate the virtual environment with python dependencies
```sh
conda env create --file=environment.yml
conda activate handobject_dataset_creator
```

Build the code using cmake:
```sh
mkdir build/ && cd build/
cmake ..
make
cd ../
ln -s build/handobject_registration .
ln -s build/scene_viewer .
```

## Recording Data

Use the `k4arecorder` included in the Azure Kinect SDK to capture hardware-synchronized RGB-D recordings.
Exemplary commands can be found [here](https://docs.microsoft.com/en-us/azure/kinect-dk/record-external-synchronized-units).

## Camera Setup and Calibration

Set up your Azure Kinect DK devices for synchronized recording as described [here](https://docs.microsoft.com/en-US/azure/Kinect-dk/multi-camera-sync).

Throughout this project, we select one camera to be the main camera and define its color sensor as the origin of our global coordinate system. All other sensor data will be transformed into this global coordinate frame.
We found the factory calibration of the sensors intrinsics as well as the extrinsics between the color and depth sensor sufficiently accurate, but this might vary with each individual camera.
Still, the relative pose between the devices (i.e. the extrinsics of the secondary camera) has to be calibrated.

The [MATLAB Stereo Camera Calibrator App](https://www.mathworks.com/help/vision/ug/stereo-camera-calibrator-app.html)
is one option to calibrate the extrinsics between the color sensors of the Azure Kinect DK devices. 
Create a short (synchronized) recording while holding a chessboard in various poses. 

Extract synchronized RGB frames from the recordings using ffmpeg
```sh
ffmpeg -i main_recording.mkv -map 0:0 -vsync 0 main_color/frame_%04d.png
ffmpeg -i sub_recording.mkv -map 0:0 -vsync 0 sub_color/frame_%04d.png
```
and select about 10-20 image pairs for calibration. Load these image pairs into the Stereo Camera Calibrator App and run the calibration.
Export the 4x4 transformation matrix containing the extrinsic parameters of the secondary camera's color sensor. 
The matrix should be stored in a simple text file similar to the [exemplary calibration file](assets/secondary_camera_color_extrinsics.npy).

## Ground Truth Recovery

### Initial Frame Labelling

Extract an initial frame from the recording by running:
```sh
scene_viewer --main path/to/main_camera.mkv --sub path/to/sub_camera.mkv --extrinsics path/to/extrinsics.txt --start_time 0 --save_first_frame
```
Adjust the `start_time` if necessary. 
Alternatively, use the `--show_pc` flag to enable the interactive point cloud visualization and press `s` to save the currently shown point cloud.
The point cloud will be stored at `pointcloud_raw_TIMESTAMP.ply`.

Create an initial tool pose guess by manually aligning the tool to the extracted point cloud using a program of your choice, e.g. [Meshlab](https://www.meshlab.net/).
Store the inital tool pose guess in a .txt file with the timestamp in the first line and the 4x4 transformation matrix in the following lines, as shown in the [example pose file](assets/initial_pose_guess.txt).
The inital guess does not have to be perfecly aligned as it will be refined via ICP in the next step.

Then, manually segment the extracted point cloud and copy all points which belong to the hand to a separate file, e.g. `recording_grasp.ply`. 
Remove all points that belong to the arm, as these points can decrease the alignment accuracy.

To recover the hand pose, first manually select a point for the wrist, each hand joint, as well as the finger tips on the outer hand surface.
Store these 16 points in a .txt file in the required [hand joint order](assets/hand_joint_label_ordering.txt),
as shown in the [exemplary hand label file](assets/hand_joint_labels.txt).

To recover the per-frame MANO hand pose parameters, run:
```sh
python3 mano_fitting.py --config config/mano_fitting.conf
```
Adjust the arguments in the provided [config file](config/mano_fitting.conf) to fit your directory structure.

### Automatic Frame Labelling
Next, Run the automatic frame labelling tool to recover the tool poses for all remaining frames. 
Adjust the required paths in the provided [config file](config/handobject_registration.conf) and run:
```sh
handobject_registration --config config/handobject_registration.conf
```
Use the `show_pc` and `show_rgb` flags to monitor the accuracy of the ICP pose recovery and detect if the pose diverges from the true pose.

Once the tool poses are recovered, extract the labelled RGB frames.
Adjust the required paths in the provided [config file](config/scene_viewer.conf) and run:
```sh
scene_viewer --config config/scene_viewer.conf
```

Last, extract object-centered patches and store the RGB image, segmentation mask, and ground truth labels in a new dataset directory.
Adjust the required paths in the provided [config file](config/extract_patches.conf) and run:
```sh
python3 extract_patches --config config/extract_patches.conf
```

## Citations

If you find this code useful for your research, please consider citing:

* the publication that this code was adapted for
```
COMING SOON.
```

* the publication it builds upon and that this code was originally developed for
```
@inproceedings{hasson20_handobjectconsist,
  title     = {Leveraging Photometric Consistency over Time for Sparsely Supervised Hand-Object Reconstruction},
  author    = {Hasson, Yana and Tekin, Bugra and Bogo, Federica and Laptev, Ivan and Pollefeys, Marc and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2020}
}
```
