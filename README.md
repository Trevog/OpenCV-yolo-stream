# OpenCV-yolo-stream
Real time object detection in (youtube) video stream:
* OpenCV with YOLOv3 detectionmethod (https://pjreddie.com/darknet/yolo/) 
* Streamlink (https://github.com/streamlink/streamlink)
<br><br>
`$  pip install opencv-python`

## Intro
This code is made for doing object detection in a video stream, then writing the the number of detected objects to an output file, every 5  seconds. This is done by making use of the OpenCV library with the YOLOv3 detectionmethod. For an introduction to opencv and yolo refer to: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/. The sourcecode from this blog, by Adrian Rosebrock, is the starting point for this repository.

## YOLO Weights & CFG:
Download the YOLOv3 weights:
 `$ wget https://pjreddie.com/media/files/yolov3.weights`
The weights are trained on the COCO dataset (http://cocodataset.org/#home)

## Example Output
![](output1.jpg)

