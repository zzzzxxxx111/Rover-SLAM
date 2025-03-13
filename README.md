# A real-time, robust and versatile visual-SLAM framework based on deep learning networks

# Prerequisites
We have tested the library in **Ubuntu 20.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.



## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
**Required at leat 3.0. Tested with OpenCV 3.4.1**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## ONNXRuntime
**Required onnxruntime-linux-x64-gpu-1.16.3**


## ROS (optional)

We provide some examples to process input of a monocular, monocular-inertial, stereo, stereo-inertial or RGB-D camera using ROS. Building these examples is optional. These have been tested with ROS Melodic under Ubuntu 18.04.

## Download Examples Folder
Download "Examples" Folder zip, and unzip in ROVER-SLAM/

## Download Dbow File
Download "voc_binary_tartan_8u_6.gz", and unzip in ROVER-SLAM/Vocabulary/

# Building Rover-SLAM library and examples

Clone the repository:
```
git clone https://github.com/zzzzxxxx111/SLslam.git
```


```
cd Rover-slam
mkdir build
cd build
cmake ..
make -j12
```


# Running 

## Euroc-monocluar:
```
./Examples/Monocular/mono_euroc  Vocabulary/voc_binary_tartan_8u_6.yml.gz Examples/Monocular/EuRoC.yaml /media/xiao/data3/slamdataset/euroc/V202 /media/xiao/data3/learning-slam/Rover-slam/Examples/Monocular/EuRoC_TimeStamps/V202.txt
```

## Euroc-monocluar-Inerial:

```
./Examples/Monocular-Inertial/mono_inertial_euroc  Vocabulary/voc_binary_tartan_8u_6.yml.gz Examples/Monocular-Inertial/EuRoC.yaml /media/xiao/data3/slamdataset/euroc/V203 media/xiao/data3/learning-slam/Rover-slam/Examples/Monocular-Inertial/EuRoC_TimeStamps/V203.txt
```



## TUM-Monocular-Inertial

```
./Examples/Monocular-Inertial/mono_inertial_tum_vi Vocabulary/voc_binary_tartan_8u_6.yml.gz Examples/Monocular-Inertial/TUM_512.yaml /media/xiao/data3/slamdataset/dataset-corridor3_512_16/mav0/cam0/data Examples/Monocular-Inertial/TUM_TimeStamps/dataset-corridor3_512.txt Examples/Monocular-Inertial/TUM_IMU/dataset-corridor3_512.txt dataset-corridor3_512_monoi
```
## Euroc -Stereo-Inertial

```
 ./Examples/Stereo-Inertial/stereo_inertial_euroc /media/xiao/data3/learning-slam/ORB_SLAM3_detailed_comments/Vocabulary/voc_binary_tartan_8u_6.yml.gz Examples/Stereo-Inertial/EuRoC.yaml /media/xiao/data3/slamdataset/euroc/V203 /media/xiao/data3/learning-slam/ORB_SLAM3_detailed_comments/Examples/Stereo/EuRoC_TimeStamps/V203.txt V203_si
```      
The rest of the operations are the same as ORB-SLAM3

Here’s an English version of the acknowledgments section for your `README` file:

---

# Acknowledgments

The completion of this project would not have been possible without the support and contributions of the following open-source projects and tools. We extend our sincere gratitude to:

1. **ORB-SLAM3**  
   

2. **AIRVO**  
  

3. **SP-Loop**  
   