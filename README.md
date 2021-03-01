# Nvidia deep stream sdk

## Requirements:

* [Docker](https://docs.docker.com/get-docker/)
* [Nvidia docker](https://github.com/NVIDIA/nvidia-docker)
* Nvidia driver installed on your pc moder. We used nvidia-driver-460  
    ```
    sudo apt install nvidia-driver-460
    sudo reboot
    nvidia-smi 
    ```


## Utils

* [license plate detector] (https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app)
* [Nvidia deep stream sdk](https://ngc.nvidia.com/catalog/containers/nvidia:deepstream)
* [Nvidia container](https://ngc.nvidia.com/catalog/containers/)
* [Nvidia Transfer Learning Toolkit](https://ngc.nvidia.com/catalog/containers/nvidia:tlt-streamanalytics)
* [Training Instance Segmentation Moduels using Mask R-CNN and Nvidia Transfer Learning Toolkit](https://developer.nvidia.com/blog/training-instance-segmentation-models-using-maskrcnn-on-the-transfer-learning-toolkit/)
* [Triton server 21.02 cuda 11.2](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel_21-02.html#rel_21-02)
* [Application fro Deepstream 5.0 /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps)

## Setup

* [Source Nvidia deep stream setup](https://ngc.nvidia.com/catalog/containers/nvidia:deepstream)

```
cd /home/visionlab/Documents/nvidia/src/

# License Plate detector
git clone https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app.git
git clone https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation.git
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

# Deep Stream SDK inference
docker pull nvcr.io/nvidia/deepstream:5.1-21.02-triton

# Transfer Learning toolkit
docker pull  nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3


# Transfler learning instance to convert models
xhost +local:docker && docker run --gpus all --name tlt --rm  -it --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/visionlab/Documents/nvidia/src/:/workspace/develop/ nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3 /bin/bash

---------- Samples
cd /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-test1
deepstream-test1-app /workspace/develop/video/parking5.h264

cd /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-test3
deepstream-test3-app file:///workspace/develop/video/parking5.mp4 file:///workspace/develop/video/parking5.mp4 file:///workspace/develop/video/parking5.mp4 file:///workspace/develop/video/parking5.mp4 

cd /opt/nvidia/deepstream/deepstream-5.1/sources/apps/sample_apps/deepstream-opencv-test
deepstream-opencv-test file:///workspace/develop/video/parking5.mp4

cd /opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_Yolo
deepstream-app -c deepstream_app_config_yoloV3.txt

----------- Python apps
Read docker part /workspace/develop/deepstream_python_apps/apps/README
cd /workspace/develop/deepstream_python_apps/deepstream_test3
python3 deepstream_test_3.py file:///workspace/develop/video/parking5.mp4 file:///workspace/develop/video/parking5.mp4  file:///workspace/develop/video/parking5.mp4  file:///workspace/develop/video/parking5.mp4 

python3 deepstream_test1_rtsp_out.py -i /workspace/develop/video/parking5.h264

# ---------- License Plate Detector Setup
cd /workspace/develop/deepstream-lpr-app
# DS5.0.1 gst-nvinfer cannot generate TRT engine for LPR model, so generate it with tlt-converter
./download_us.sh
tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 models/LP/LPR/us_lprnet_baseline18_deployable.etlt -t fp16 -e models/LP/LPR/lpr_us_onnx_b16.engine
./download_ch.sh
tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 models/LP/LPR/ch_lprnet_baseline18_deployable.etlt -t fp16 -e models/LP/LPR/lpr_ch_onnx_b16.engine
cd deepstram_lpr_app
cp dict_us.txt dict.txt
#cp dict_ch.txt dict.txt

# Examples
cd /workspace/examples
jupyter notebook --ip 0.0.0.0 --allow-root



# Deep Stream SDK Triton 
xhost +local:docker && docker run --gpus all --name deepstream -it --rm  --device=/dev/video0  --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/visionlab/Documents/nvidia/src/:/workspace/develop/ nvcr.io/nvidia/deepstream:5.1-21.02-triton /bin/bash
cd /workspace/develop/deepstream_lpr_app/
make
./deepstream-lpr-app <1:US car plate model|2: Chinese car plate model> \
         <1: output as h264 file| 2:fakesink 3:display output> <0:ROI disable|1:ROI enable> \
         <input mp4 file name> ... <input mp4 file name> <output file name>
cd deepstream-lpr-app && ./deepstream-lpr-app 1 3 1 ../parking5.mp4 ../parking5.mp4 ../parking5_out.mp4


# Useful
# Get nvidia deep stream sdk x86
# https://ngc.nvidia.com/catalog/containers/nvidia:deepstream/tags
# All availab sdk 
#docker pull nvcr.io/nvidia/deepstream:5.1-21.02-base
#docker pull nvcr.io/nvidia/deepstream:5.1-21.02-iot
#docker pull nvcr.io/nvidia/deepstream:5.1-21.02-samples
#docker pull nvcr.io/nvidia/deepstream:5.1-21.02-devel
#apt update && apt install build-essential
ffmpeg -i parking5.mp4 -vcodec h264 parking5.h264
```

```

```
