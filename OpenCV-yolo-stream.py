# run in command prompt (no output files)
# python OpenCV-yolo-stream.py --yolo yolo-coco --url https://youtu.be/1EiC9bvVGnk

# run in command prompt (with output files)
# python OpenCV-yolo-stream.py --yolo yolo-coco --url https://youtu.be/1EiC9bvVGnk --output output/ouput_videosteam.avi --data output/CSV/data_videosteam.csv 

# JacksonHole streams https://youtu.be/RZWzyQuFxgE & https://youtu.be/1EiC9bvVGnk

# import the necessary packages
import numpy as np
import pandas as pd
import argparse
import time
import datetime
import cv2
import os
import pafy
import streamlink

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True,
    help="video url")
ap.add_argument("-p", "--period", type=float, default=5,
    help="execution period")
ap.add_argument("-o", "--output", required=False,
    help="path to output video")
ap.add_argument("-d", "--data", required=False,
    help="path to output csv")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to yolov weights, cfg and coco directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.55,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# set execution period 
period = args["period"]

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("Initializing...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

url = args["url"]

vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
streams = streamlink.streams(url)

# set initial parameters
writer = None
(W, H) = (None, None)
starttime=time.time()
frame_ind = 0
obj = np.zeros((1000,7))
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    framedatetime = datetime.datetime.now()
    framedatetime = framedatetime.strftime('%Y%m%d%H%M%S')
    cap = cv2.VideoCapture(streams["best"].url)
    (grabbed,frame) = cap.read()
    #(grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    #set initial objects to 0
    persons = 0
    cars = 0
    trucks = 0
    busses = 0
    # ensure at least one detection exists
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # check for specific objects
            if ("{}".format(LABELS[classIDs[i]]) == "person") or ("{}".format(LABELS[classIDs[i]]) == "car") or ("{}".format(LABELS[classIDs[i]]) == "truck") or ("{}".format(LABELS[classIDs[i]]) == "bus"):
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                # count specific objects
                if "{}".format(LABELS[classIDs[i]]) == "person":
                  persons+=1
                if "{}".format(LABELS[classIDs[i]]) == "car":
                  cars+=1
                if "{}".format(LABELS[classIDs[i]]) == "truck":
                  trucks+=1
                if "{}".format(LABELS[classIDs[i]]) == "bus":
                  busses+=1
    # construct a tuple of information we will be displaying on the frame
    info = [
        ("Busses", busses),
        ("Trucks", trucks),
        ("Cars", cars),
        ("Persons", persons),   
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 30) + 30)),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    # check if video output directory is given
    if args["output"] is not None:
        # check if the video writer is None
        if writer is None:
          # initialize our video writer
          fourcc = cv2.VideoWriter_fourcc(*"MJPG")
          writer = cv2.VideoWriter(args["output"], fourcc, 30,
              (frame.shape[1], frame.shape[0]), True)
        #write the output frame to disk
        writer.write(frame)

    # check if data output directory is given
    if args["data"] is not None:
        # write number of detections to array
        obj[frame_ind][0] = int(frame_ind+1)
        obj[frame_ind][1] = len(idxs)
        obj[frame_ind][2] = int(persons)
        obj[frame_ind][3] = int(cars)
        obj[frame_ind][4] = int(trucks)
        obj[frame_ind][5] = int(busses)
        obj[frame_ind][6] = int(framedatetime)
        # save obj as csv every 10 frames
        if frame_ind % 10 == 0:
          obj_df = pd.DataFrame(obj)
          obj_df.columns = ['Frame', 'Objects', 'Persons', 'Cars', 'Trucks', 'Busses', 'DateTime']
          obj_df.to_csv(args["data"])

    # print object detection info 
    print("frame: {:.0f}".format(int(frame_ind+1)), "   datetime:", str(framedatetime))
    print("             persons: {:.0f}".format(int(persons)))
    print("                cars: {:.0f}".format(int(cars)))
    print("              trucks: {:.0f}".format(int(trucks)))
    print("              busses: {:.0f}".format(int(busses)))

    # wait, if period is not over jet
    time.sleep(period - ((time.time() - starttime) % period))
    # show the output frame
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    frameR = cv2.resize(frame, (960, 540))
    cv2.imshow("Frame", frameR)

    frame_ind += 1

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# check if data output directory is given
if args["data"] is not None:
    #save obj as csv 
    obj_df = pd.DataFrame(obj)
    obj_df.columns = ['Frame', 'Objects', 'Persons', 'Cars', 'Trucks', 'Busses', 'DateTime']
    obj_df.to_csv(args["data"]) 

cv2.destroyAllWindows()


